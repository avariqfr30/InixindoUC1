"""Financial analysis and OSINT-backed budget estimation."""
from __future__ import annotations

import concurrent.futures
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ollama import Client
from pydantic import BaseModel, Field

from .proposal_shared import LLM_MODEL, SPIRIT_OF_AI_RULES, logger
from .research import Researcher

class FinancialSchema(BaseModel):
    revenue_idr: Optional[int] = Field(None, description="Total revenue in IDR")
    profit_idr: Optional[int] = Field(None, description="Total profit in IDR")
    project_budget_idr: Optional[int] = Field(None, description="Project budget in IDR")
    source_quote: str = Field("", description="Exact quote from text")


class FinancialAnalyzer:
    def __init__(self, ollama_client: Client):
        self.ollama = ollama_client

    DEFAULT_DURATION_MONTHS = {
        "diagnostic": 2.0, "strategic": 3.0, "transformation": 6.0, "implementation": 5.0,
    }

    BASE_MONTHLY_DELIVERY_RATE = {
        "diagnostic": 45_000_000, "strategic": 60_000_000, "transformation": 85_000_000, "implementation": 95_000_000,
    }

    def _extract_financials_with_llm(self, company_name: str, markdown_text: str) -> Dict[str, Any]:
        """Uses Ollama and Pydantic to strictly extract financial data from scraped markdown text."""
        if not markdown_text.strip():
            return {"revenue_idr": None, "profit_idr": None, "project_budget_idr": None, "source_quote": ""}
            
        prompt = f"""
        You are a financial analyst. Read the following text about {company_name}.
        Extract the most recent financial figures available. 
        
        TEXT:
        {markdown_text}
        
        Respond ONLY with a valid JSON object using this exact schema. If a value is not mentioned, use null.
        {{
            "revenue_idr": <number or null>,
            "profit_idr": <number or null>,
            "project_budget_idr": <number or null>,
            "source_quote": "<copy the exact sentence where you found the numbers>"
        }}
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed_dict = json.loads(match.group(0)) if match else json.loads(raw_text)
            
            # Pydantic strictly validates the types and structure
            valid_data = FinancialSchema.model_validate(parsed_dict)
            return valid_data.model_dump()
            
        except Exception as e:
            logger.error(f"Pydantic/LLM Extraction failed: {e}")
            return {"revenue_idr": None, "profit_idr": None, "project_budget_idr": None, "source_quote": ""}

    @staticmethod
    def _has_signal(text: str, candidate: str) -> bool:
        value = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
        if not value: return False
        pattern = re.escape(value)
        if " " not in value and value.isalpha(): pattern = rf"\b{pattern}\b"
        return bool(re.search(pattern, (text or "").lower()))

    @classmethod
    def _ai_scope_summary(cls, *values: Any) -> Dict[str, Any]:
        combined = re.sub(r"\s+", " ", " ".join(str(value or "") for value in values)).strip().lower()
        strong_hits = [token for token in (SPIRIT_OF_AI_RULES.get("strong_trigger_keywords") or []) if cls._has_signal(combined, token)]
        support_hits = [token for token in (SPIRIT_OF_AI_RULES.get("supporting_signals") or []) if token not in strong_hits and cls._has_signal(combined, token)]
        return {
            "enabled": bool(strong_hits) or len(support_hits) >= 2,
            "strong_hits": strong_hits[:8],
            "support_hits": support_hits[:10],
        }

    @classmethod
    def _ai_pricing_profile(cls, project_goal: str, objective: str, notes: str, frameworks: str, scope_context: str = "") -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or "", scope_context or ""]).lower()
        ai_scope = cls._ai_scope_summary(project_goal, objective, notes, frameworks, scope_context)
        if not ai_scope["enabled"]:
            return {"enabled": False, "level": "terkendali", "multiplier": 1.0, "drivers": [], "driver_labels": []}

        driver_config = SPIRIT_OF_AI_RULES.get("pricing_driver_terms") or {}
        driver_labels = {
            "data_readiness": "kesiapan data/model", "model_uncertainty": "validasi solusi/model",
            "architecture_constraints": "kendala arsitektur dan keamanan", "governance_overhead": "governance dan compliance",
            "change_enablement": "enablement dan adopsi organisasi",
        }
        driver_weights = {"data_readiness": 0.10, "model_uncertainty": 0.10, "architecture_constraints": 0.08, "governance_overhead": 0.10, "change_enablement": 0.09}

        multiplier = 1.0
        active_drivers: List[str] = []
        for driver, terms in driver_config.items():
            if any(cls._has_signal(combined, term) for term in (terms or [])):
                multiplier += driver_weights.get(driver, 0.06)
                active_drivers.append(driver)

        if not active_drivers:
            active_drivers = ["data_readiness", "governance_overhead", "change_enablement"]
            multiplier += 0.16

        multiplier = max(1.0, min(1.8, multiplier))
        level = "tinggi" if multiplier >= 1.45 else "menengah" if multiplier >= 1.18 else "terkendali"
        return {"enabled": True, "level": level, "multiplier": multiplier, "drivers": active_drivers, "driver_labels": [driver_labels.get(d, d) for d in active_drivers]}

    @staticmethod
    def _format_idr(amount: int) -> str:
        return "Rp " + f"{max(0, int(amount)):,}".replace(",", ".")

    @staticmethod
    def _parse_number(raw: str) -> Optional[float]:
        value = (raw or "").strip()
        if not value: return None
        if "." in value and "," in value: value = value.replace(".", "").replace(",", ".")
        elif "," in value and "." not in value: value = value.replace(",", ".")
        elif "." in value and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value): value = value.replace(".", "")
        try: return float(value)
        except ValueError: return None

    @classmethod
    def _extract_financial_values(cls, text: str) -> List[int]:
        if not text: return []
        pattern = re.compile(r"(?i)(?P<rp>rp\.?\s*)?(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<unit>triliun|miliar|juta|ribu)?")
        multiplier = {"triliun": 1_000_000_000_000, "miliar": 1_000_000_000, "juta": 1_000_000, "ribu": 1_000}
        values: List[int] = []
        for match in pattern.finditer(text):
            has_rp = bool(match.group("rp"))
            unit = (match.group("unit") or "").lower()
            base = cls._parse_number(match.group("num") or "")
            if base is None or (not has_rp and not unit): continue
            amount = int(base * multiplier.get(unit, 1))
            if amount > 0 and (amount >= 1_000_000 or unit): values.append(amount)
        return values

    @classmethod
    def _duration_to_months(cls, timeline: str) -> Optional[float]:
        text = re.sub(r"\([^)]*\)", " ", (timeline or "").lower()).strip()
        if not text: return None
        patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(tahun|thn|year|years)", 12.0),
            (r"(\d+(?:[.,]\d+)?)\s*(bulan|bln|month|months)", 1.0),
            (r"(\d+(?:[.,]\d+)?)\s*(minggu|week|weeks)", 1.0 / 4.345),
            (r"(\d+(?:[.,]\d+)?)\s*(hari|day|days)", 1.0 / 30.0),
        ]
        months = 0.0
        found = False
        for pattern, factor in patterns:
            for match in re.finditer(pattern, text):
                val = cls._parse_number(match.group(1))
                if val is not None:
                    months += val * factor
                    found = True
        if found: return max(months, 0.5)
        single_number = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if single_number:
            val = cls._parse_number(single_number.group(1))
            if val is not None: return max(val, 0.5)
        return None

    @classmethod
    def _duration_months_or_default(cls, timeline: str, project_type: str) -> float:
        return cls._duration_to_months(timeline) or cls.DEFAULT_DURATION_MONTHS.get((project_type or "").strip().lower(), 4.0)

    @classmethod
    def _complexity_profile(cls, project_goal: str, objective: str, notes: str, frameworks: str, scope_context: str = "") -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or "", scope_context or ""]).lower()
        if not combined.strip(): return {"level": "moderat", "multiplier": 1.0, "signal_count": 0}

        high_complexity = {"integrasi": 0.10, "core banking": 0.14, "migrasi": 0.10, "nasional": 0.06, "enterprise": 0.06, "regulasi": 0.08, "security": 0.06}
        medium_complexity = {"dashboard": 0.03, "governance": 0.04, "cloud": 0.04, "api": 0.04, "audit": 0.04, "change management": 0.05}

        multiplier = 1.0
        signal_count = 0
        for token, boost in {**high_complexity, **medium_complexity}.items():
            if token in combined:
                multiplier += boost
                signal_count += 1

        framework_tokens = [p.strip() for p in re.split(r"[,;/]| dan ", frameworks or "") if p.strip()]
        if framework_tokens: multiplier += min(0.12, max(0, len(framework_tokens) - 1) * 0.03)

        scope_multiplier = cls._scope_pricing_multiplier(scope_context, combined)
        multiplier *= scope_multiplier["multiplier"]
        signal_count += scope_multiplier["signal_count"]

        ai_profile = cls._ai_pricing_profile(project_goal, objective, notes, frameworks, scope_context)
        if ai_profile["enabled"]:
            multiplier *= ai_profile["multiplier"]
            signal_count += len(ai_profile["drivers"])

        multiplier = max(0.9, min(1.9, multiplier))
        level = "tinggi" if multiplier >= 1.45 else "menengah" if multiplier >= 1.15 else "terkendali"
        return {"level": level, "multiplier": multiplier, "signal_count": signal_count}

    @classmethod
    def _scope_pricing_multiplier(cls, scope_context: str, combined_text: str = "") -> Dict[str, Any]:
        text = " ".join([scope_context or "", combined_text or ""]).lower()
        if not text.strip():
            return {"multiplier": 1.0, "signal_count": 0, "profile": "tidak disebutkan"}

        broad_terms = [
            "implementasi", "integrasi", "migrasi", "go-live", "golive", "deployment",
            "pengadaan", "lisensi", "pelatihan", "rollout", "change management",
        ]
        boundary_terms = [
            "out-of-scope", "di luar cakupan", "di luar lingkup", "tidak termasuk",
            "exclude", "batasan", "asesmen", "workshop", "roadmap", "rekomendasi",
        ]
        broad_hits = sum(1 for token in broad_terms if token in text)
        boundary_hits = sum(1 for token in boundary_terms if token in text)
        if broad_hits >= 3:
            return {"multiplier": 1.0 + min(0.30, broad_hits * 0.05), "signal_count": broad_hits, "profile": "luas"}
        if boundary_hits >= 2 and broad_hits <= 1:
            return {"multiplier": 0.88, "signal_count": boundary_hits, "profile": "terbatas"}
        return {"multiplier": 1.0, "signal_count": broad_hits + boundary_hits, "profile": "moderat"}

    @classmethod
    def build_silent_scope_context(cls, timeline: str = "", project_type: str = "", service_type: str = "", project_goal: str = "", objective: str = "", notes: str = "", frameworks: str = "", commercial_context: str = "") -> str:
        parts = [
            f"Durasi: {timeline}" if str(timeline or "").strip() else "",
            f"Jenis proyek: {project_type}" if str(project_type or "").strip() else "",
            f"Jenis layanan: {service_type}" if str(service_type or "").strip() else "",
            f"Klasifikasi kebutuhan: {project_goal}" if str(project_goal or "").strip() else "",
            f"Konteks pekerjaan: {objective}" if str(objective or "").strip() else "",
            f"Permasalahan dan kebutuhan: {notes}" if str(notes or "").strip() else "",
            f"Kerangka kerja: {frameworks}" if str(frameworks or "").strip() else "",
            f"Standar komersial internal: {commercial_context}" if str(commercial_context or "").strip() else "",
        ]
        return " | ".join(part for part in parts if part)

    @classmethod
    def _project_effort_baseline(cls, timeline: str, project_type: str, service_type: str, project_goal: str, objective: str, notes: str, frameworks: str, scope_context: str = "") -> Dict[str, Any]:
        months = cls._duration_months_or_default(timeline, project_type)
        monthly_rate = cls.BASE_MONTHLY_DELIVERY_RATE.get((project_type or "").strip().lower(), 70_000_000)
        service_multiplier = {"training": 0.80, "konsultan": 1.00, "training dan konsultan": 1.12}.get((service_type or "").strip().lower(), 1.0)
        complexity = cls._complexity_profile(project_goal, objective, notes, frameworks, scope_context)
        ai_pricing = cls._ai_pricing_profile(project_goal, objective, notes, frameworks, scope_context)

        active_dims = sum(1 for item in [objective, notes, project_goal, frameworks, scope_context] if str(item or "").strip())
        breadth_mult = 1.0 + min(0.18, max(0, active_dims - 2) * 0.04)

        effort_base = monthly_rate * months * service_multiplier * complexity["multiplier"] * breadth_mult
        if ai_pricing["enabled"]: effort_base *= max(1.0, ai_pricing["multiplier"] * 0.92)
        return {
            "months": months, "monthly_rate": int(monthly_rate), "complexity": complexity,
            "ai_pricing": ai_pricing, "effort_base": int(max(120_000_000, min(9_000_000_000, effort_base)))
        }

    @staticmethod
    def _bounded_calibration(value: int, anchor: int, lower_ratio: float, upper_ratio: float) -> int:
        return max(int(anchor * lower_ratio), min(int(anchor * upper_ratio), max(value, anchor)))

    @classmethod
    def _dynamic_budget_from_osint(cls, client_name: str, finance_snippets: List[Dict[str, Any]], benchmark_snippets: List[Dict[str, Any]], timeline: str = "", project_type: str = "", service_type: str = "", project_goal: str = "", objective: str = "", notes: str = "", frameworks: str = "", llm_financial_data: Optional[Dict[str, Any]] = None, scope_context: str = "") -> Dict[str, Any]:
        finance_text = " ".join([str(i.get("title", "")) + " " + str(i.get("snippet", "")) for i in (finance_snippets or [])])
        benchmark_text = " ".join([str(i.get("title", "")) + " " + str(i.get("snippet", "")) for i in (benchmark_snippets or [])])

        finance_values = sorted(cls._extract_financial_values(finance_text))
        benchmark_values = sorted(cls._extract_financial_values(benchmark_text))
        if not scope_context:
            scope_context = cls.build_silent_scope_context(timeline, project_type, service_type, project_goal, objective, notes, frameworks)
        effort_profile = cls._project_effort_baseline(timeline, project_type, service_type, project_goal, objective, notes, frameworks, scope_context)
        effort_base = effort_profile["effort_base"]

        llm_rev = (llm_financial_data or {}).get("revenue_idr")
        if llm_rev and isinstance(llm_rev, (int, float)) and llm_rev > 0:
            finance_median = int(llm_rev)
            financial_base = cls._bounded_calibration(int(max(120_000_000, min(6_000_000_000, finance_median * 0.0015))), effort_base, 0.70, 1.80)
        elif finance_values:
            finance_median = finance_values[len(finance_values) // 2]
            financial_base = cls._bounded_calibration(int(max(120_000_000, min(6_000_000_000, finance_median * 0.0015))), effort_base, 0.70, 1.80)
        else:
            finance_median, financial_base = None, effort_base

        benchmark_median = benchmark_values[len(benchmark_values) // 2] if benchmark_values else None
        market_base = cls._bounded_calibration(int(max(80_000_000, min(6_000_000_000, benchmark_median))), effort_base, 0.75, 1.60) if benchmark_median else effort_base

        if finance_median and benchmark_median: base_price = int((effort_base * 0.65) + (market_base * 0.25) + (financial_base * 0.10))
        elif benchmark_median: base_price = int((effort_base * 0.78) + (market_base * 0.22))
        elif finance_median: base_price = int((effort_base * 0.88) + (financial_base * 0.12))
        else: base_price = effort_base

        calibration_cap = max(effort_base, int(financial_base * 1.75)) if finance_median else effort_base
        adjusted_base = int(max(120_000_000, min(9_000_000_000, min(base_price, calibration_cap))))

        basic = int(adjusted_base * 0.72)
        return {
            "analysis": f"Estimasi untuk {client_name} dihitung berdasarkan durasi {effort_profile['months']} bulan, ruang lingkup tersirat dari konteks kerja, dan tingkat kompleksitas {effort_profile['complexity']['level']}.",
            "options": [
                {"tier": "Basic", "price": cls._format_idr(basic)},
                {"tier": "Standard", "price": cls._format_idr(max(basic + 40_000_000, adjusted_base))},
                {"tier": "Enterprise", "price": cls._format_idr(max(adjusted_base + 80_000_000, int(adjusted_base * 1.65)))},
            ],
        }

    def suggest_budget(self, client_name: str, timeline: str = "", project_type: str = "", service_type: str = "", project_goal: str = "", objective: str = "", notes: str = "", frameworks: str = "", commercial_context: str = "", pricing_mode: str = "demo", scope_context: str = "") -> Dict[str, Any]:
        year = datetime.now().year
        
        # Parallel OSINT Fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_finance = executor.submit(Researcher.search, f'"{client_name}" laporan keuangan OR pendapatan {year-1} OR {year}', 6, "year")
            future_bench = executor.submit(Researcher.search, f'estimasi biaya proyek {project_type} {service_type} {timeline} Indonesia', 6, "year")
        
        finance_snippets = Researcher._filter_recent_entity_results(future_finance.result(), client_name, max_age_years=3)
        benchmark_snippets = Researcher._sort_by_recency(future_bench.result())[:5]
        
        # Deep Scrape Financials
        llm_financial_data = None
        if finance_snippets and finance_snippets[0].get("link"):
            markdown = Researcher.fetch_full_markdown(finance_snippets[0]["link"])
            if markdown: llm_financial_data = self._extract_financials_with_llm(client_name, markdown)
        
        if not scope_context:
            scope_context = self.build_silent_scope_context(timeline, project_type, service_type, project_goal, objective, notes, frameworks, commercial_context)
        return self._dynamic_budget_from_osint(client_name, finance_snippets, benchmark_snippets, timeline, project_type, service_type, project_goal, objective, notes, frameworks, llm_financial_data, scope_context)
