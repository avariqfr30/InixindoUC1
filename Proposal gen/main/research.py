"""External research, OSINT fetching, and research-cache helpers."""
from __future__ import annotations

import concurrent.futures
import functools
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import diskcache as dc
from ollama import Client
from pydantic import BaseModel, Field

from .proposal_shared import *

# Initialize ultra-fast disk caching for OSINT (survives server restarts)
osint_cache_dir = Path(APP_STATE_DB_PATH).parent / '.osint_cache'
osint_cache = dc.Cache(str(osint_cache_dir))

import functools

def smart_osint_cache(expire_success=86400, expire_empty=3600, ignore_empty=True, inject_llm=False):
    """
    Production-grade decorator replacing manual cache management.
    - ignore_empty: Bypasses cache if the function returns falsy (prevents caching API failures).
    - bifurcated TTLs: Caches empty results for a shorter time if ignore_empty=False.
    - inject_llm: Automatically hashes the LLM_MODEL into the key to respect model upgrades.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key_parts = [func.__name__]
            if inject_llm:
                key_parts.append(str(LLM_MODEL))
                
            for arg in args:
                if isinstance(arg, str):
                    key_parts.append(" ".join(arg.lower().split()))
                else:
                    key_parts.append(str(arg))
                    
            for k, v in sorted(kwargs.items()):
                val = " ".join(v.lower().split()) if isinstance(v, str) else str(v)
                key_parts.append(f"{k}={val}")
                
            cache_key = ":".join(key_parts)
            
            cached = osint_cache.get(cache_key)
            if cached is not None:
                return cached
                
            result = func(*args, **kwargs)
            
            if not result and ignore_empty:
                return result
                
            ttl = expire_success if result else expire_empty
            osint_cache.set(cache_key, result, expire=ttl)
            return result
        return wrapper
    return decorator

class InsightSchema(BaseModel):
    insight: str = Field(description="The extracted insight in Indonesian. 'NOT_FOUND' if missing.")


# Public research helper (Serper & LLM Extractors).
class Researcher:
    CONTEXT_STOPWORDS = {
        "yang", "dengan", "untuk", "dari", "pada", "dan", "atau", "agar", "sebagai",
        "dalam", "akan", "lebih", "oleh", "serta", "suatu", "para", "bagi", "atas",
        "this", "that", "from", "with", "into", "their", "client", "proposal",
        "project", "service", "mode", "response", "canvassing", "kak", "kerangka",
        "acuan", "kerja", "jenis", "proyek", "proposal", "perusahaan", "klien",
        "pekerjaan", "kebutuhan", "inisiatif", "prioritas", "terukur",
    }

    @staticmethod
    def _has_serper_key() -> bool:
        key = (SERPER_API_KEY or "").strip()
        if not key:
            return False
        placeholder_keys = {"YOUR_SERPER_API_KEY", "YOUR_SERPER", "SERPER_API"}
        return key not in placeholder_keys

    @staticmethod
    @smart_osint_cache(expire_success=43200, ignore_empty=True)
    def fetch_full_markdown(url: str) -> str:
        """Fetches the clean markdown text of any URL using Jina Reader."""
        if not url: return ""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(jina_url, headers=headers, timeout=12)
            if response.status_code == 200:
                return response.text[:6000] 
            return ""
        except Exception as e:
            logger.warning(f"Failed to fetch full markdown for {url}: {e}")
            return ""

    @staticmethod
    @smart_osint_cache(expire_success=43200, ignore_empty=True, inject_llm=True)
    def extract_insight_with_llm(url: str, extraction_goal: str) -> str:
        """Universal Deep Scraper: Reads a URL and extracts a specific qualitative insight via Pydantic/LLM."""
        markdown_text = Researcher.fetch_full_markdown(url)
        if not markdown_text:
            return ""
            
        prompt = f"""
        You are an expert business researcher. Read the following source text.
        Your goal is to extract: {extraction_goal}
        
        SOURCE TEXT:
        {markdown_text}
        
        Respond ONLY with a valid JSON object using this schema. If the information is not present, use "NOT_FOUND".
        {{
            "insight": "<concise professional summary in Indonesian>"
        }}
        """
        
        try:
            client = Client(host=OLLAMA_HOST)
            res = client.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed_dict = json.loads(match.group(0)) if match else json.loads(raw_text)
            
            # Use Pydantic to strictly validate and coerce the dictionary
            data = InsightSchema.model_validate(parsed_dict)
            
            if "NOT_FOUND" in data.insight.upper() or not data.insight:
                return ""
            return Researcher._clean_osint_fact(data.insight)
        except Exception as e:
            logger.warning(f"Insight extraction failed for {url}: {e}")
            return ""

    @staticmethod
    def _clean_osint_fact(text: str) -> str:
        cleaned = str(text or "").replace("\xa0", " ").replace("…", " ")
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\.{3,}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|,;:")
        return cleaned

    @staticmethod
    def _format_source_evidence_item(item: Dict[str, Any], fact_override: str = "", index: int = 1) -> str:
        title = str(item.get("title", "") or "Sumber tanpa judul").strip()
        link = str(item.get("link", "") or "-").strip()
        fact = Researcher._clean_osint_fact(str(fact_override or item.get("snippet", "") or ""))
        if not fact:
            return ""
        source_name = Researcher._source_name(link)
        citation = f"({source_name}, {Researcher._citation_year(item)})"
        return f"Sumber eksternal {max(1, int(index or 1))}: fakta={fact} | sumber={title} | url={link} | sitasi_apa={citation}"

    @staticmethod
    def _combine_summary_and_evidence(summary: str, evidence_lines: List[str], fallback: str = "") -> str:
        parts: List[str] = []
        cleaned_summary = str(summary or "").strip()
        if cleaned_summary:
            parts.append(cleaned_summary)

        deduped_lines: List[str] = []
        seen: Set[str] = set()
        for raw_line in evidence_lines or []:
            line = str(raw_line or "").strip()
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_lines.append(line)

        if deduped_lines:
            parts.append("\n".join(deduped_lines))

        if not parts:
            return fallback
        return "\n".join(parts)

    @staticmethod
    @smart_osint_cache(expire_success=86400, expire_empty=3600, ignore_empty=False)
    def search(query: str, limit: int = 5, recency_bucket: str = "") -> List[Dict[str, Any]]:
        """General web search using Serper.dev"""
        if not Researcher._has_serper_key():
            return []
        
        url = "https://google.serper.dev/search"
        payload_data = {"q": query, "gl": "id", "num": limit}
        recency_map = {"week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
        if recency_bucket in recency_map:
            payload_data["tbs"] = recency_map[recency_bucket]
        
        try:
            response = requests.post(
                url, 
                headers={'X-API-KEY': (SERPER_API_KEY or "").strip(), 'Content-Type': 'application/json'}, 
                data=json.dumps(payload_data), 
                timeout=8
            )
            response.raise_for_status()
            return response.json().get('organic', [])
        except requests.RequestException as e:
            logger.warning(f"Serper API Error | query='{query[:60]}...' | error={str(e)[:100]}")
            return []

    @staticmethod
    def _dedupe_results(items: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for item in items or []:
            link = str(item.get("link") or "").strip().lower()
            title = Researcher._normalize_text(str(item.get("title") or ""))
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max(1, int(limit or 10)):
                break
        return deduped

    @staticmethod
    def _search_multi(queries: List[str], limit_per_query: int = 5, recency_bucket: str = "year", max_results: int = 10) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for query in queries or []:
            cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
            if not cleaned:
                continue
            results.extend(Researcher.search(cleaned, limit=limit_per_query, recency_bucket=recency_bucket))
        return Researcher._sort_by_recency(Researcher._dedupe_results(results, limit=max_results))

    @staticmethod
    def _is_trusted_regulatory_source(link: str) -> bool:
        host = urlparse(str(link or "")).netloc.lower().replace("www.", "")
        trusted_exact = {"iso.org", "ietf.org", "ojk.go.id", "bi.go.id", "bssn.go.id", "kominfo.go.id"}
        return any(host == domain or host.endswith(f".{domain}") for domain in trusted_exact) or host.endswith(".go.id")

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _infer_industry_from_context(cls, client_name: str, ai_context: str = "", regulations: str = "") -> str:
        combined = " ".join([client_name or "", ai_context or "", regulations or ""]).lower()
        rules = [
            ("Perbankan", ["bank", "bri", "bca", "mandiri", "bni", "btn", "fintech", "kredit", "asset management"]),
            ("Telekomunikasi", ["telkom", "telkomsel", "indosat", "xl", "axiata", "operator"]),
            ("Energi & Utilitas", ["pertamina", "pln", "energi", "listrik", "oil", "gas"]),
            ("Ritel & E-Commerce", ["tokopedia", "e-commerce", "retail", "marketplace", "goto", "alfamart", "alfamidi"]),
            ("Transportasi & Aviasi", ["garuda", "aviation", "airline", "logistik", "transport", "bluebird"]),
            ("Pemerintah & BUMN", ["kementerian", "dinas", "pemprov", "pemkab", "bumn", "spbe"]),
            ("Manufaktur & Tambang", ["manufaktur", "adaro", "antam", "vale", "smelter", "mining"]),
        ]
        for label, tokens in rules:
            if any(token in combined for token in tokens):
                return label
        return "Lintas Industri"

    @staticmethod
    def _industry_query_terms(industry: str) -> List[str]:
        terms_map = {
            "Perbankan": ["manajemen investasi", "operasional investasi", "kepatuhan POJK", "service reliability"],
            "Telekomunikasi": ["service assurance", "network reliability", "customer experience"],
            "Energi & Utilitas": ["operational reliability", "governance", "asset integrity"],
            "Ritel & E-Commerce": ["omnichannel", "operational governance", "customer experience"],
            "Transportasi & Aviasi": ["operational control", "service reliability", "customer experience"],
            "Pemerintah & BUMN": ["tata kelola", "akuntabilitas", "audit trail", "layanan publik"],
            "Manufaktur & Tambang": ["production governance", "operational reliability", "safety compliance"],
        }
        return terms_map.get(industry, ["tata kelola", "operational excellence", "business value"])

    @classmethod
    def _context_terms(cls, text: str, excluded_terms: Optional[List[str]] = None, max_terms: int = 8) -> List[str]:
        excluded: Set[str] = set()
        for item in (excluded_terms or []):
            normalized = cls._normalize_text(item)
            if not normalized:
                continue
            excluded.add(normalized)
            excluded.update(normalized.split())
        tokens: List[str] = []
        seen: Set[str] = set()
        for token in re.findall(r"[A-Za-z]{4,}", (text or "").lower()):
            normalized = cls._normalize_text(token)
            if (
                not normalized
                or normalized in seen
                or normalized in cls.CONTEXT_STOPWORDS
                or normalized in excluded
            ):
                continue
            seen.add(normalized)
            tokens.append(token)
            if len(tokens) >= max_terms:
                break
        return tokens

    @staticmethod
    def _framework_terms(regulations: str, max_items: int = 4) -> List[str]:
        cleaned: List[str] = []
        seen: Set[str] = set()
        for part in re.split(r"[\n,;|/]+", str(regulations or "")):
            value = re.sub(r"\s+", " ", part).strip(" -.:")
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _focus_phrase(ai_context: str, max_words: int = 10) -> str:
        source = str(ai_context or "").replace("\r", "\n")
        line = re.split(r"[\n.;]+", source, maxsplit=1)[0]
        line = re.sub(r"\s+", " ", line).strip(" -|,:")
        if not line:
            return ""
        words = line.split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words).strip(" ,;:-")

    @staticmethod
    def _query_or_clause(terms: List[str], max_items: int = 4) -> str:
        cleaned: List[str] = []
        seen: Set[str] = set()
        for term in terms or []:
            value = re.sub(r"\s+", " ", str(term or "").strip())
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(f'"{value}"' if " " in value else value)
            if len(cleaned) >= max_items:
                break
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "(" + " OR ".join(cleaned) + ")"

    @classmethod
    def _build_query_plan(cls, client_name: str, ai_context: str = "", regulations: str = "") -> Dict[str, Any]:
        entity_terms = cls._entity_tokens(client_name)
        focus_phrase = cls._focus_phrase(ai_context)
        framework_terms = cls._framework_terms(regulations)
        industry = cls._infer_industry_from_context(client_name, ai_context, regulations)
        industry_terms = cls._industry_query_terms(industry)
        focus_terms = cls._context_terms(
            ai_context,
            excluded_terms=[client_name, focus_phrase, *framework_terms, *entity_terms],
            max_terms=8,
        )
        return {
            "focus_phrase": focus_phrase,
            "focus_terms": focus_terms,
            "framework_terms": framework_terms,
            "industry": industry,
            "industry_terms": industry_terms,
        }

    @staticmethod
    def _entity_tokens(entity_name: str) -> List[str]:
        legal_tokens = {"pt", "cv", "tbk", "inc", "ltd", "co", "corp", "persero", "company"}
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", (entity_name or "").lower())
            if len(token) >= 3 and token not in legal_tokens
        ]
        ordered = []
        seen = set()
        for token in tokens:
            if token in seen: continue
            ordered.append(token)
            seen.add(token)
        return ordered

    @staticmethod
    def _is_entity_match(item: Dict[str, Any], entity_name: str, strict: bool = False) -> bool:
        if not entity_name: return True
        merged = " ".join([str(item.get("title", "")), str(item.get("snippet", "")), str(item.get("link", ""))])
        normalized_merged = Researcher._normalize_text(merged)
        phrase = Researcher._normalize_text(entity_name)
        tokens = Researcher._entity_tokens(entity_name)

        if phrase and phrase in normalized_merged: return True
        if not tokens: return False

        merged_tokens = set(normalized_merged.split())
        hits = sum(1 for token in tokens if token in merged_tokens)
        if strict: return hits == len(tokens)
        if len(tokens) == 1: return hits == 1
        if len(tokens) == 2: return hits == 2
        return hits >= (len(tokens) - 1)

    @staticmethod
    def _extract_month(text: str) -> Optional[int]:
        if not text: return None
        month_map = {
            "jan": 1, "januari": 1, "feb": 2, "februari": 2, "mar": 3, "maret": 3,
            "apr": 4, "april": 4, "mei": 5, "may": 5, "jun": 6, "juni": 6,
            "jul": 7, "juli": 7, "agu": 8, "agustus": 8, "aug": 8, "sep": 9,
            "sept": 9, "september": 9, "okt": 10, "oct": 10, "oktober": 10,
            "nov": 11, "november": 11, "des": 12, "dec": 12, "desember": 12,
        }
        lowered = (text or "").lower()
        for key, month in month_map.items():
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                return month
        return None

    @staticmethod
    def _extract_day(text: str) -> Optional[int]:
        if not text: return None
        match = re.search(r"\b([0-2]?\d|3[01])\b", text)
        if not match: return None
        day = int(match.group(1))
        if 1 <= day <= 31: return day
        return None

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        if not text: return None
        years = re.findall(r'\b(20\d{2})\b', text)
        if not years: return None
        return max(int(y) for y in years)

    @staticmethod
    def _published_sort_key(item: Dict[str, Any]) -> int:
        merged = " ".join([str(item.get("date", "")), str(item.get("title", "")), str(item.get("snippet", ""))])
        year = Researcher._extract_year(merged) or 0
        month = Researcher._extract_month(merged) or 1
        day = Researcher._extract_day(str(item.get("date", ""))) or 1
        return (year * 10_000) + (month * 100) + day

    @staticmethod
    def _sort_by_recency(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items or [], key=Researcher._published_sort_key, reverse=True)

    @staticmethod
    def _is_recent(item: Dict[str, Any], max_age_years: int = 2) -> bool:
        merged = " ".join([str(item.get('date', '')), str(item.get('snippet', '')), str(item.get('title', ''))])
        year = Researcher._extract_year(merged)
        if year is None: return True
        return year >= (datetime.now().year - max_age_years)

    @staticmethod
    def _filter_recent_entity_results(items: List[Dict[str, Any]], entity_name: str = "", max_age_years: int = 2, strict_entity: bool = False) -> List[Dict[str, Any]]:
        filtered = []
        for item in (items or []):
            if not Researcher._is_recent(item, max_age_years=max_age_years): continue
            if entity_name and not Researcher._is_entity_match(item, entity_name, strict=strict_entity): continue
            filtered.append(item)
        return Researcher._sort_by_recency(filtered)

    @staticmethod
    def _source_name(link: str) -> str:
        try:
            host = urlparse(link or "").netloc.lower().strip().replace("www.", "")
            return host or "sumber daring"
        except Exception:
            return "sumber daring"

    @staticmethod
    def _citation_year(item: Dict[str, Any]) -> str:
        merged = " ".join([str(item.get('date', '')), str(item.get('snippet', '')), str(item.get('title', ''))])
        year = Researcher._extract_year(merged)
        return str(year) if year else "n.d."

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]], label: str, fallback: str) -> str:
        if not items: return f"{fallback} (sumber daring, n.d.)"
        lines = []
        for i, item in enumerate(items, start=1):
            title = item.get('title', 'Sumber tanpa judul')
            snippet = Researcher._clean_osint_fact(str(item.get('snippet', '') or ''))
            link = item.get('link', '-')
            if not snippet: continue
            source_name = Researcher._source_name(link)
            citation = f"({source_name}, {Researcher._citation_year(item)})"
            lines.append(f"Sumber eksternal {i}: fakta={snippet} | sumber={title} | url={link} | sitasi_apa={citation}")
        return "\n".join(lines) if lines else f"{fallback} (sumber daring, n.d.)"

    # =========================================================
    # UPGRADED OSINT METHODS: USING DEEP SCRAPING + PYDANTIC
    # =========================================================

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_latest_client_news(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:3], *plan.get("framework_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" berita inovasi OR transformasi digital OR inisiatif strategis'
        if context_clause:
            query = f'"{client_name}" {context_clause} berita inovasi OR transformasi digital OR inisiatif strategis'
        query = f"{query} {current_year}"
        queries = [
            query,
            f'"{client_name}" "{plan.get("industry")}" strategi bisnis inisiatif {current_year}',
            f'"{client_name}" {context_clause} roadmap program prioritas {current_year}' if context_clause else "",
        ]
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="month", max_results=10)
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=2)
        
        # Deep Scrape the #1 top news hit
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            focus_hint = plan.get("focus_phrase") or Researcher._query_or_clause(plan.get("framework_terms", []), max_items=2)
            goal = (
                f"What is the company's latest major strategic initiative, digital transformation, or business innovation"
                f"{f' related to {focus_hint}' if focus_hint else ''}?"
            )
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                source = Researcher._source_name(top_link)
                summary = f"Berdasarkan inisiatif strategis terbaru dari liputan {source}: {insight}"
                evidence_lines = [Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)]
                corroborating = [
                    line for line in (
                        Researcher._format_source_evidence_item(item, index=idx)
                        for idx, item in enumerate(filtered[1:3], start=2)
                    )
                    if line
                ]
                evidence_lines.extend(corroborating)
                return Researcher._combine_summary_and_evidence(summary, evidence_lines, fallback="")
                
        # Fallback to standard snippets
        cleaned = [i for i in filtered if len(re.findall(r"\S+", i.get('snippet',''))) > 5]
        return Researcher._format_evidence(cleaned[:3] if cleaned else filtered[:3], label="OSINT_NEWS", fallback="")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_ai_posture(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [*plan.get("focus_terms", [])[:3], *plan.get("framework_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" AI OR artificial intelligence OR generative AI OR machine learning OR data platform'
        if context_clause:
            query = f'{query} {context_clause}'
        query = f"{query} {current_year}"
        queries = [
            query,
            f'"{client_name}" data analytics automation digital platform {current_year}',
            f'"{client_name}" {context_clause} AI data governance {current_year}' if context_clause else "",
        ]
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="year", max_results=10)
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=3)
        
        # Deep Scrape the #1 AI posture link
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            focus_hint = plan.get("focus_phrase") or Researcher._query_or_clause(plan.get("focus_terms", []), max_items=3)
            goal = (
                f"What is {client_name}'s current maturity level or stated plans regarding Artificial Intelligence, Data, or Machine Learning"
                f"{f' for initiatives related to {focus_hint}' if focus_hint else ''}?"
            )
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                source = Researcher._source_name(top_link)
                summary = f"Kesiapan AI & Data {client_name} (via {source}): {insight}"
                evidence_lines = [Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)]
                corroborating = [
                    line for line in (
                        Researcher._format_source_evidence_item(item, index=idx)
                        for idx, item in enumerate(filtered[1:3], start=2)
                    )
                    if line
                ]
                evidence_lines.extend(corroborating)
                return Researcher._combine_summary_and_evidence(
                    summary,
                    evidence_lines,
                    fallback="Data publik terkait AI posture klien terbatas.",
                )
                
        return Researcher._format_evidence(filtered[:3], label="OSINT_AI", fallback="Data publik terkait AI posture klien terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_firm_values_approach(firm_name: str) -> str:
        query = f'"{firm_name}" misi OR visi OR nilai OR pendekatan OR metodologi OR prinsip'
        res = Researcher.search(query, limit=5, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, firm_name, max_age_years=5)
        
        # Deep Scrape the firm's approach from their top hit
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            goal = f"What are the core professional values, mission statement, or unique working methodologies of {firm_name}?"
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                summary = f"Pendekatan dan Nilai Inti {firm_name}: {insight}"
                evidence_line = Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)
                return Researcher._combine_summary_and_evidence(summary, [evidence_line], fallback=summary)
        
        return "Pendekatan menekankan delivery berkualitas tinggi, kolaborasi erat, dan hasil bisnis terukur."

    # Standard Snippet Methods (Kept for speed)
    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_entity_profile(entity_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        strict_entity = Researcher._normalize_text(entity_name) == Researcher._normalize_text(WRITER_FIRM_NAME)
        plan = Researcher._build_query_plan(entity_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [*plan.get("industry_terms", [])[:2], *plan.get("framework_terms", [])[:2], *plan.get("focus_terms", [])[:2]],
            max_items=4,
        )
        query = f'"{entity_name}" profil perusahaan OR "tentang kami" OR layanan utama'
        if context_clause:
            query = f'"{entity_name}" {context_clause} profil perusahaan OR "tentang kami" OR layanan utama'
        query = f"{query} {current_year}"
        queries = [
            query,
            f'"{entity_name}" annual report profil bisnis layanan utama {current_year}',
            f'"{entity_name}" {context_clause} profil bisnis {current_year}' if context_clause else "",
        ]
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="year", max_results=10)
        filtered = Researcher._filter_recent_entity_results(res, entity_name, max_age_years=3, strict_entity=strict_entity)
        return Researcher._format_evidence(filtered[:3], label="OSINT_PROFILE", fallback="Data profil terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_track_record(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:3], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" pencapaian OR kinerja OR penghargaan OR implementasi OR transformasi'
        if context_clause:
            query = f'"{client_name}" {context_clause} pencapaian OR kinerja OR penghargaan OR implementasi OR transformasi'
        queries = [
            query,
            f'"{client_name}" kinerja bisnis transformasi operasional',
            f'"{client_name}" {context_clause} pencapaian implementasi' if context_clause else "",
        ]
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="year", max_results=10)
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=3)
        return Researcher._format_evidence(filtered[:3], label="OSINT_TRACK", fallback="Track record terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_writer_collaboration(client_name: str, writer_firm_name: str = WRITER_FIRM_NAME, ai_context: str = "", regulations: str = "") -> str:
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("framework_terms", [])[:2], *plan.get("focus_terms", [])[:2]],
            max_items=4,
        )
        query = f'"{writer_firm_name}" "{client_name}" kerja sama OR proyek OR konsultasi'
        if context_clause:
            query = f'"{writer_firm_name}" "{client_name}" {context_clause} kerja sama OR proyek OR konsultasi'
        queries = [
            query,
            f'"{writer_firm_name}" "{client_name}" pelatihan OR sertifikasi OR pendampingan',
        ]
        if context_clause:
            queries.append(f'"{writer_firm_name}" "{client_name}" {context_clause}')
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="year", max_results=10)
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=6)
        return Researcher._format_evidence(filtered[:2], label="OSINT_COLLAB", fallback="Belum ditemukan bukti publik kolaborasi.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_regulatory_data(regulations_string: str, ai_context: str = "", client_name: str = "") -> str:
        if not regulations_string: return "Tidak ada regulasi spesifik dari input user."
        plan = Researcher._build_query_plan(client_name, ai_context, regulations_string)
        framework_clause = Researcher._query_or_clause(plan.get("framework_terms", []), max_items=4)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=4,
        )
        query = f'Ringkasan implementasi standar {framework_clause or regulations_string.replace(",", " OR ")}'
        if context_clause:
            query = f"{query} {context_clause}"
        queries = [
            f"{query} site:.go.id",
            f"{query} site:iso.org",
            f"{query} site:ojk.go.id OR site:bi.go.id" if "bank" in Researcher._normalize_text(client_name + ' ' + ai_context) else "",
            f"{query} site:bssn.go.id OR site:kominfo.go.id",
        ]
        res = Researcher._search_multi(queries, limit_per_query=5, recency_bucket="year", max_results=10)
        recent = [i for i in Researcher._sort_by_recency(res) if Researcher._is_recent(i, max_age_years=5)]
        trusted = [i for i in recent if Researcher._is_trusted_regulatory_source(str(i.get("link", "")))]
        return Researcher._format_evidence((trusted or recent)[:3], label="OSINT_REG", fallback="Data regulasi terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_firm_certifications(firm_name: str) -> str:
        query = f'"{firm_name}" ISO OR sertifikasi OR akreditasi OR certification'
        res = Researcher.search(query, limit=5, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, firm_name, max_age_years=5)
        certs = list(set(re.findall(r'ISO[\s\-]?\d{4,5}|[A-Z]{2,4}\s+\d{3,5}', " ".join(i.get('snippet','') for i in filtered))))
        if certs: return f"Sertifikasi: {', '.join(certs[:4])}. Lihat website resmi untuk daftar lengkap."
        return f"Kredensial dan sertifikasi {firm_name} tersedia di saluran publik resmi."

    @staticmethod
    def build_comprehensive_firm_profile(firm_name: str, office_location: str = "") -> Dict[str, str]:
        # Using concurrent.futures to fetch all firm OSINT at the EXACT same time
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_values = executor.submit(Researcher.get_firm_values_approach, firm_name)
            future_certs = executor.submit(Researcher.get_firm_certifications, firm_name)
            
        return {
            "values_approach": future_values.result(),
            "certifications": future_certs.result(),
            "team_expertise": f"Tim profesional {firm_name} memiliki pengalaman di berbagai domain strategis.",
            "portfolio_scale": f"Portofolio {firm_name} mencakup berbagai klien enterprise.",
            "key_contacts": f"Hubungi {firm_name} melalui saluran resmi untuk koordinasi detail.",
            "accolades": f"{firm_name} terus membangun reputasi melalui proyek-proyek tepat guna.",
        }


