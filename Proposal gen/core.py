"""
Core business logic for API adapters, vector databases, OSINT capabilities,
and document generation.
"""

import os
import io
import re
import json
import logging
import requests
import pandas as pd
import chromadb
from chromadb.config import Settings
import concurrent.futures
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Any, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageStat
from difflib import SequenceMatcher

from sqlalchemy import create_engine
import markdown
from bs4 import BeautifulSoup

from docx import Document
from docx.image.exceptions import UnrecognizedImageError
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from ollama import Client 
from chromadb.utils import embedding_functions

from config import (
    SERPER_API_KEY, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, DB_URI,
    WRITER_FIRM_NAME, DEFAULT_COLOR, UNIVERSAL_STRUCTURE, 
    PERSONAS, PROPOSAL_SYSTEM_PROMPT, DATA_MAPPING, 
    DEMO_MODE, FIRM_API_URL, API_AUTH_TOKEN, MOCK_FIRM_STANDARDS, MOCK_FIRM_PROFILE
)

logger = logging.getLogger(__name__)


# =====================================================================
# FIRM API ADAPTER
# =====================================================================
class FirmAPIClient:
    def __init__(self) -> None:
        self.demo_mode = DEMO_MODE
        self.base_url = FIRM_API_URL
        self.headers = {"Authorization": f"Bearer {API_AUTH_TOKEN}"}

    def get_project_standards(self, project_type: str) -> Dict[str, str]:
        if self.demo_mode:
            logger.info(f"[DEMO] Using Mock Data for type: {project_type}")
            return MOCK_FIRM_STANDARDS.get(project_type, MOCK_FIRM_STANDARDS.get("Implementation"))
        try:
            res = requests.get(f"{self.base_url}/standards/{project_type}", headers=self.headers, timeout=5)
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            logger.error(f"Internal API Error: {e}")
            return {"methodology": "TBD", "team": "TBD", "commercial": "TBD"}

    def get_firm_profile(self) -> Dict[str, str]:
        if self.demo_mode:
            return MOCK_FIRM_PROFILE
        try:
            res = requests.get(f"{self.base_url}/firm-profile", headers=self.headers, timeout=5)
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            logger.error(f"Internal API Error: {e}")
            return {"contact_info": "Kantor Pusat Terdaftar", "portfolio_highlights": "Penyedia Solusi IT"}


# =====================================================================
# KNOWLEDGE BASE & VECTOR DB
# =====================================================================
class KnowledgeBase:
    def __init__(self, db_uri: str) -> None:
        self.engine = create_engine(db_uri)
        self.chroma = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings", model_name=EMBED_MODEL
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", embedding_function=self.embed_fn
        )
        self.df: Optional[pd.DataFrame] = None
        self.refresh_data()

    def refresh_data(self) -> bool:
        try:
            self.df = pd.read_sql("SELECT * FROM projects", self.engine)
        except Exception:
            if not os.path.exists("db.csv"):
                return False
            raw_df = pd.read_csv("db.csv")
            raw_df.columns = [c.strip() for c in raw_df.columns]
            rename_dict = {v: k for k, v in DATA_MAPPING.items()}
            raw_df.rename(columns=rename_dict, inplace=True)
            raw_df.to_sql("projects", self.engine, index=False, if_exists='replace')
            self.df = raw_df
            
        existing_ids = set(self.collection.get()['ids'])
        new_ids_map = {str(idx): row for idx, row in self.df.iterrows()}
        new_ids_set = set(new_ids_map.keys())
        
        ids_to_delete = list(existing_ids - new_ids_set)
        ids_to_add = list(new_ids_set - existing_ids)
        
        if ids_to_delete: 
            self.collection.delete(ids_to_delete)
            
        if ids_to_add:
            for i in range(0, len(ids_to_add), 500):
                batch_ids = ids_to_add[i:i + 500]
                docs = [" | ".join([f"{col}: {val}" for col, val in new_ids_map[b].items()]) for b in batch_ids]
                metas = [new_ids_map[b].astype(str).to_dict() for b in batch_ids]
                self.collection.add(documents=docs, metadatas=metas, ids=batch_ids)
                
        return True

    def get_exact_context(self, entity: str, topic: str, budget: Optional[str] = None) -> str:
        if self.df is None or self.df.empty:
            return "No data."
        try:
            match = self.df[(self.df['entity'] == entity) & (self.df['topic'] == topic)]
            if budget and not match.empty:
                match = match[match['budget'] == budget]
            if not match.empty:
                return "".join([f"- {k.capitalize()}: {v}\n" for k, v in match.iloc[0].to_dict().items()])
            return "No data."
        except Exception:
            return ""

    def query(self, client: str, project: str, context_keywords: str = "") -> str:
        try:
            res = self.collection.query(query_texts=[f"{project} for {client} {context_keywords}"], n_results=2)
            if res['documents'] and len(res['documents'][0]) > 0:
                return "\n".join(res['documents'][0])
            return ""
        except Exception:
            return ""


# =====================================================================
# OSINT RESEARCHER (SERPER.DEV INTEGRATION)
# =====================================================================
class Researcher:
    @staticmethod
    @lru_cache(maxsize=256)
    def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """General web search using Serper.dev"""
        if "YOUR_SERPER" in SERPER_API_KEY:
            return []
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "id", "num": limit})
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=8)
            response.raise_for_status()
            return response.json().get('organic', [])
        except requests.RequestException as e:
            logger.warning(f"Serper API Error: {e}")
            return []
    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        if not text:
            return None
        years = re.findall(r'\b(20\d{2})\b', text)
        if not years:
            return None
        return max(int(y) for y in years)

    @staticmethod
    def _is_recent(item: Dict[str, Any], max_age_years: int = 2) -> bool:
        merged = " ".join([
            str(item.get('date', '')),
            str(item.get('snippet', '')),
            str(item.get('title', '')),
        ])
        year = Researcher._extract_year(merged)
        if year is None:
            return True
        return year >= (datetime.now().year - max_age_years)

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]], label: str, fallback: str) -> str:
        if not items:
            return f"[{label}] {fallback}"
        lines = []
        for i, item in enumerate(items, start=1):
            title = item.get('title', 'Sumber tanpa judul')
            snippet = (item.get('snippet', '') or '').strip()
            link = item.get('link', '-')
            date = item.get('date', '-')
            if not snippet:
                continue
            lines.append(f"[{label} #{i}] {title} | date={date} | source={link} | fakta={snippet}")
        return "\n".join(lines) if lines else f"[{label}] {fallback}"


    @staticmethod
    @lru_cache(maxsize=128)
    def get_entity_profile(entity_name: str) -> str:
        res = Researcher.search(f'"{entity_name}" profil perusahaan OR "tentang kami" -saham -loker', limit=6)
        filtered = [i for i in res if Researcher._is_recent(i, max_age_years=3)]
        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_PROFILE",
            fallback=f"Data profil terbaru untuk {entity_name} terbatas; gunakan informasi umum yang terverifikasi saja."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_latest_client_news(client_name: str) -> str:
        current_year = datetime.now().year
        prev_year = current_year - 1
        res = Researcher.search(
            f'"{client_name}" berita inovasi OR transformasi digital {current_year} OR {prev_year}',
            limit=8
        )
        filtered = [i for i in res if Researcher._is_recent(i, max_age_years=2)]
        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_NEWS",
            fallback=f"Berita terbaru {client_name} tidak cukup kuat; jangan membuat klaim spesifik tanpa bukti."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_regulatory_data(regulations_string: str) -> str:
        if not regulations_string:
            return "[OSINT_REG] Tidak ada regulasi spesifik dari input user."

        query = f'Ringkasan implementasi standar {regulations_string.replace(",", " OR ")} site:.go.id OR site:iso.org'
        res = Researcher.search(query, limit=8)
        filtered = [i for i in res if Researcher._is_recent(i, max_age_years=5)]
        return Researcher._format_evidence(
            filtered[:5],
            label="OSINT_REG",
            fallback=f"Data regulasi untuk {regulations_string} terbatas; nyatakan asumsi dan batasan data secara eksplisit."
        )


# =====================================================================
# FINANCIAL ANALYZER (SMART PRICING)
# =====================================================================
class FinancialAnalyzer:
    def __init__(self, ollama_client: Client):
        self.ollama = ollama_client

    def suggest_budget(self, client_name: str) -> Dict[str, Any]:
        snippets = Researcher.search(f'"{client_name}" laporan keuangan OR "pendapatan" OR "pendanaan" OR "aset" 2023 OR 2024', limit=5)
        context = "\n".join([s.get('snippet', '') for s in snippets]) if snippets else "Tidak ada data finansial yang dipublikasikan secara terbuka."

        prompt = f"""
        Menganalisa kekuatan finansial perusahaan: {client_name}.
        Data OSINT: {context}

        Berdasarkan data di atas, estimasikan kapasitas finansial mereka dan berikan 3 opsi estimasi budget proyek TI/Konsultasi.
        FORMAT WAJIB JSON murni tanpa markdown, tanpa teks tambahan:
        {{
            "analysis": "Ringkasan 1 kalimat kekuatan finansial berdasarkan data (atau sebutkan estimasi jika data terbatas).",
            "options": [
                {{"tier": "Basic", "price": "Rp 100.000.000"}},
                {{"tier": "Standard", "price": "Rp 350.000.000"}},
                {{"tier": "Enterprise", "price": "Rp 800.000.000"}}
            ]
        }}
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'system', 'content': 'You output strictly valid JSON.'}, {'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"Financial Analyzer Error: {e}")
            return {
                "analysis": "Data OSINT terbatas, menggunakan standar estimasi B2B.",
                "options": [
                    {"tier": "Basic", "price": "Rp 150.000.000"},
                    {"tier": "Standard", "price": "Rp 300.000.000"},
                    {"tier": "Enterprise", "price": "Rp 750.000.000"}
                ]
            }

class LogoManager:
    @staticmethod
    def _create_fallback_logo(client_name: str) -> io.BytesIO:
        initials = "".join([w[0] for w in re.findall(r"[A-Za-z0-9]+", client_name)[:3]]).upper() or "CL"
        canvas = Image.new('RGB', (320, 320), color=(236, 242, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((16, 16, 304, 304), radius=36, outline=(37, 99, 235), width=8)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 110)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), initials, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(initials, font=font)
        draw.text(((320 - w) / 2, (320 - h) / 2 - 6), initials, fill=(15, 23, 42), font=font)
        out = io.BytesIO()
        canvas.save(out, format='PNG')
        out.seek(0)
        return out

    @staticmethod
    def get_logo_and_color(client_name: str) -> Tuple[Optional[io.BytesIO], Tuple[int, int, int]]:
        if "YOUR_SERPER" in SERPER_API_KEY:
            return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR
        try:
            url = "https://google.serper.dev/images"
            payload = json.dumps({"q": f"{client_name} corporate logo png transparent", "num": 3})
            headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
            res = requests.post(url, headers=headers, data=payload, timeout=8).json()
            
            if 'images' in res and res['images']:
                for item in res['images']:
                    try:
                        img_resp = requests.get(item['imageUrl'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if img_resp.status_code == 200:
                            stream = io.BytesIO(img_resp.content)
                            img = Image.open(stream)
                            
                            # Ensure DOCX-compatible format by converting any source format (e.g. WEBP/SVG fallback) to PNG.
                            if img.mode in ("RGBA", "LA", "P"):
                                normalized = Image.new("RGBA", img.size, (255, 255, 255, 0))
                                normalized.paste(img, (0, 0), img if img.mode in ("RGBA", "LA") else None)
                                img = normalized.convert('RGB')
                            else:
                                img = img.convert('RGB')

                            img.thumbnail((150, 150))
                            dom_color = list(map(int, ImageStat.Stat(img).mean[:3]))

                            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                            if luminance > 120:
                                factor = 120 / luminance
                                dom_color = [max(0, min(255, int(c * factor))) for c in dom_color]

                            # Return normalized PNG bytes instead of original content to avoid UnrecognizedImageError.
                            png_stream = io.BytesIO()
                            img.save(png_stream, format='PNG')
                            png_stream.seek(0)
                            return png_stream, tuple(dom_color)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Logo Retrieval Error: {e}")
        return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR


# =====================================================================
# RENDERING ENGINES
# =====================================================================
class StyleEngine:
    @staticmethod
    def apply_document_styles(doc: Document) -> None:
        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        pf = style.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.15
        pf.space_after = Pt(8) 
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)

class ChartEngine:
    @staticmethod
    def _get_plt_color(theme_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(c/255 for c in theme_color)

    @staticmethod
    def create_gantt_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = data_str.split('|')
            if len(parts) == 3:
                title_str, unit_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else:
                title_str, unit_str, raw_data = "Timeline", "Waktu", data_str
            tasks = []
            for p in raw_data.split(';'):
                t_parts = p.split(',')
                if len(t_parts) >= 3:
                    tasks.append({"task": t_parts[0].strip(), "start": float(re.sub(r'[^\d.]', '', t_parts[1])), "dur": float(re.sub(r'[^\d.]', '', t_parts[2]))})
            if not tasks: return None
            tasks = tasks[::-1] 
            
            fig, ax = plt.subplots(figsize=(8.5, max(4, len(tasks)*0.8)))
            for i, task in enumerate(tasks):
                rect = patches.FancyBboxPatch((task['start'], i-0.3), task['dur'], 0.6, boxstyle="round,pad=0.02", ec="#ffffff", fc=ChartEngine._get_plt_color(theme_color), alpha=0.9, lw=1.5)
                ax.add_patch(rect)
                ax.text(task['start'] + (task['dur'] / 2), i, f"{task['dur']:g} {unit_str}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
            names = [t['task'] for t in tasks]
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            max_x = max([t['start'] + t['dur'] for t in tasks])
            ax.set_xlim(0, max_x + (max_x * 0.1))
            ax.set_ylim(-0.6, len(names)-0.4)
            
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

class DocumentBuilder:
    @staticmethod
    def _append_text_run(paragraph, text: str, bold: bool = False, italic: bool = False) -> None:
        cleaned = re.sub(r'\s+', ' ', text or '').strip()
        if not cleaned:
            return

        # Keep spacing between runs stable so marker/text and inline formatting stay aligned.
        if paragraph.runs:
            last_text = paragraph.runs[-1].text or ""
            if last_text and not last_text.endswith((" ", "\t", "\n", "(", "[", "/")) and not cleaned.startswith((".", ",", ";", ":", ")", "]", "%")):
                cleaned = " " + cleaned

        run = paragraph.add_run(cleaned)
        run.bold = bold
        run.italic = italic

    @staticmethod
    def parse_html_to_docx(doc: Document, html_content: str, theme_color: Tuple[int, int, int]) -> None:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.children:
            if element.name is None: continue
            if element.name in ['h1', 'h2', 'h3']:
                level = int(element.name[1])
                p = doc.add_heading(element.get_text().strip(), level=level)
                if level == 1:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.color.rgb = RGBColor(*theme_color)
                    run.font.name = 'Arial'
                    run.bold = True
            elif element.name == 'p':
                p = doc.add_paragraph()
                DocumentBuilder._process_inline_html(p, element)
            elif element.name in ['ul', 'ol']:
                # Render list markers manually to prevent DOCX auto-number continuation across chapters.
                direct_items = element.find_all('li', recursive=False)
                for idx, li in enumerate(direct_items, start=1):
                    # Skip empty markers (prevents orphan numbers like "3." on its own line).
                    if not li.get_text(" ", strip=True):
                        continue
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Cm(0.63)
                    p.paragraph_format.first_line_indent = Cm(-0.38)
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(4)
                    marker = f"{idx}.\t" if element.name == 'ol' else "•\t"
                    p.add_run(marker).bold = True
                    DocumentBuilder._process_inline_html(p, li)
            elif element.name == 'table':
                rows = element.find_all('tr')
                if not rows: continue
                max_cols = max([len(r.find_all(['td', 'th'])) for r in rows])
                table = doc.add_table(rows=len(rows), cols=max_cols)
                table.style = 'Table Grid'
                for i, row in enumerate(rows):
                    cols = row.find_all(['td', 'th'])
                    for j, col in enumerate(cols):
                        if j < max_cols:
                            cell = table.cell(i, j)
                            cell._element.clear_content()
                            p = cell.add_paragraph()
                            DocumentBuilder._process_inline_html(p, col)

    @staticmethod
    def _process_inline_html(paragraph, element):
        for child in element.children:
            if child.name in ['strong', 'b']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), bold=True)
            elif child.name in ['em', 'i']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), italic=True)
            elif child.name == 'br':
                paragraph.add_run("\n")
            elif child.name is None:
                DocumentBuilder._append_text_run(paragraph, str(child))
            else:
                DocumentBuilder._process_inline_html(paragraph, child)

    @staticmethod
    def process_content(doc: Document, raw_text: str, theme_color: Tuple[int, int, int], chapter_title: str) -> None:
        clean_lines = []
        in_table = False
        for line in raw_text.split('\n'):
            line = line.strip()
            if line.startswith('[[GANTT:') and line.endswith(']]'):
                data = line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(6))
                continue
            if line.startswith('|'):
                if not in_table and clean_lines and clean_lines[-1] != "":
                    clean_lines.append("")
                in_table = True
            else:
                in_table = False
            clean_lines.append(line)
            
        html = markdown.markdown("\n".join(clean_lines), extensions=['tables'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)


class ProposalGenerator:
    def __init__(self, kb_instance: KnowledgeBase) -> None:
        self.ollama = Client(host=OLLAMA_HOST)
        self.kb = kb_instance
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.firm_api = FirmAPIClient()
        self._research_cache: Dict[str, Dict[str, str]] = {}
        self._proposal_contract_cache: Dict[str, str] = {}
        self._chapter_context_cache: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _target_words(chap: Dict[str, Any]) -> int:
        m = re.search(r'Target:\s*(\d+)\s*words', chap.get('length_intent', ''), re.IGNORECASE)
        return int(m.group(1)) if m else 700

    @staticmethod
    def _max_words(chap: Dict[str, Any]) -> int:
        return int(ProposalGenerator._target_words(chap) * 1.3)

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r'\b\w+\b', text))

    @staticmethod
    def _cache_key(*parts: Any) -> str:
        return "||".join([str(p).strip().lower() for p in parts])

    @staticmethod
    def _cache_put(cache: Dict[str, Any], key: str, value: Any, max_size: int = 128) -> None:
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_size:
            cache.pop(next(iter(cache)))

    def _resolve_chapters(self, chapter_id: Optional[str]) -> List[Dict[str, Any]]:
        normalized_id = (chapter_id or "").strip()
        if not normalized_id or normalized_id.lower() in {"all", "semua"}:
            return UNIVERSAL_STRUCTURE

        selected = [chap for chap in UNIVERSAL_STRUCTURE if chap["id"] == normalized_id]
        if selected:
            return selected

        normalized = normalized_id.lower()
        selected = [chap for chap in UNIVERSAL_STRUCTURE if chap["title"].strip().lower() == normalized]
        if selected:
            return selected

        raise ValueError(f"Unknown chapter_id: {normalized_id}")

    def build_preview_outline(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chapter_id = (data or {}).get("chapter_id")
        try:
            chapters = self._resolve_chapters(chapter_id)
        except ValueError:
            chapters = UNIVERSAL_STRUCTURE

        client = (data or {}).get("nama_perusahaan", "Klien")
        objective = (data or {}).get("konteks_organisasi", "").strip() or "tujuan proyek belum diisi"
        issues = (data or {}).get("permasalahan", "").strip() or "pain points belum diisi"
        need_type = (data or {}).get("klasifikasi_kebutuhan", "").strip() or "belum dipilih"
        project_type = (data or {}).get("jenis_proyek", "").strip() or "belum dipilih"
        service_type = (data or {}).get("jenis_proposal", "").strip() or "belum dipilih"
        frameworks = (data or {}).get("potensi_framework", "").strip() or "belum dipilih"
        timeline = (data or {}).get("estimasi_waktu", "").strip() or "belum ditentukan"
        budget = (data or {}).get("estimasi_biaya", "").strip() or "belum ditentukan"

        preview_map = {
            "c_1": f"Menetapkan konteks organisasi {client} dan objektif inisiatif: {objective}.",
            "c_2": f"Mengurai kebutuhan dan akar masalah klien berdasarkan pain points: {issues}.",
            "c_3": f"Mengkategorikan kebutuhan ke {need_type} lalu memvalidasi jenis proyek {project_type}.",
            "c_4": f"Menautkan kebutuhan klien dengan framework/regulasi utama: {frameworks}.",
            "c_5": f"Menjelaskan pemilihan metodologi delivery untuk layanan {service_type}.",
            "c_6": "Mendetailkan target state, output, dan deliverable solusi yang dapat dieksekusi.",
            "c_7": f"Menyusun rencana fase, milestone, dan deliverable berdasarkan durasi {timeline}.",
            "c_8": "Merumuskan governance proyek: forum keputusan, eskalasi isu, dan quality gate.",
            "c_9": f"Menetapkan struktur tim dan kapabilitas yang dibutuhkan untuk model {service_type}.",
            "c_10": f"Mendefinisikan model pembiayaan, termin pembayaran, dan batasan scope dengan estimasi {budget}.",
            "c_closing": f"Menutup proposal dengan apresiasi kemitraan, kontak resmi {WRITER_FIRM_NAME}, dan langkah tindak lanjut bersama {client}.",
        }

        return [
            {
                "id": chap["id"],
                "title": chap["title"],
                "preview": preview_map.get(chap["id"], "Ringkasan konten bab akan disesuaikan dengan konteks klien."),
                "subsections": chap["subs"],
            }
            for chap in chapters
        ]

    def _get_research_bundle(self, base_client: str, regulations: str) -> Dict[str, str]:
        key = self._cache_key("research", base_client, regulations)
        cached = self._research_cache.get(key)
        if cached:
            return cached

        futures = {
            "profile": self.io_pool.submit(Researcher.get_entity_profile, base_client),
            "news": self.io_pool.submit(Researcher.get_latest_client_news, base_client),
            "regulations": self.io_pool.submit(Researcher.get_regulatory_data, regulations)
        }
        try:
            bundle = {
                "profile": futures["profile"].result(timeout=8),
                "news": futures["news"].result(timeout=8),
                "regulations": futures["regulations"].result(timeout=8),
            }
        except Exception:
            bundle = {
                "profile": f"[OSINT_PROFILE] Data profil terbaru {base_client} terbatas.",
                "news": f"[OSINT_NEWS] Data berita terbaru {base_client} terbatas.",
                "regulations": "[OSINT_REG] Data regulasi terbatas."
            }

        self._cache_put(self._research_cache, key, bundle, max_size=96)
        return bundle

    def _build_proposal_contract(
        self,
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        selected_chapters: List[Dict[str, Any]],
        research_bundle: Dict[str, str],
        firm_data: Dict[str, str]
    ) -> str:
        cache_key = self._cache_key(
            "contract", client, project, budget, service_type, project_goal,
            project_type, timeline, notes, regulations, "|".join([c["id"] for c in selected_chapters])
        )
        cached = self._proposal_contract_cache.get(cache_key)
        if cached:
            return cached

        chapter_titles = ", ".join([c["title"] for c in selected_chapters])
        prompt = f"""
        Buat "Proposal Contract" ringkas untuk menjaga kualitas dan koherensi lintas bab.
        Konteks:
        - Klien: {client}
        - Inisiatif: {project}
        - Service Type: {service_type}
        - Jenis Proyek: {project_type}
        - Kebutuhan: {project_goal}
        - Durasi: {timeline}
        - Estimasi Biaya: {budget}
        - Pain Points: {notes}
        - Framework: {regulations}
        - Bab yang ditulis: {chapter_titles}
        - Baseline Metodologi: {firm_data.get('methodology', '')}
        - Baseline Team: {firm_data.get('team', '')}
        - Baseline Commercial: {firm_data.get('commercial', '')}
        - Profil OSINT: {research_bundle.get('profile', '')}
        - Berita OSINT: {research_bundle.get('news', '')}

        OUTPUT WAJIB (tanpa markdown code block, <= 220 kata):
        1) Narasi Inti (1-2 kalimat)
        2) Terminologi Kanonis (maks 6 istilah)
        3) Prinsip Konsistensi Antarbab (maks 5 butir)
        4) Larangan Gaya Tulis (maks 3 butir)
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_ctx': 16384, 'num_predict': 700, 'temperature': 0.15}
            )
            contract = (res.get('message', {}).get('content', '') or '').strip()
        except Exception:
            contract = ""

        if not contract:
            contract = (
                "Narasi Inti: Proposal harus menjawab kebutuhan bisnis klien secara konkret, terukur, dan eksekutabel.\n"
                "Terminologi Kanonis: deliverable, milestone, target state, governance, quality gate, risiko.\n"
                "Prinsip Konsistensi Antarbab: istilah konsisten, alur masalah-ke-solusi jelas, "
                "timeline sinkron dengan deliverable, tata kelola tegas, hindari repetisi.\n"
                "Larangan Gaya Tulis: filler generik, klaim tanpa dasar, paragraf tanpa tindakan."
            )

        self._cache_put(self._proposal_contract_cache, cache_key, contract, max_size=96)
        return contract

    def _fetch_chapter_context(
        self,
        chap: Dict[str, Any],
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        firm_data: Dict[str, str],
        firm_profile: Dict[str, str],
        research_bundle: Dict[str, str],
        proposal_contract: str
    ) -> Dict[str, Any]:
        try:
            global_data = research_bundle.get('profile', '')
            client_news = research_bundle.get('news', '')
            regulation_data = research_bundle.get('regulations', '')

            ctx_key = self._cache_key("chapter_ctx", client, project, budget, chap.get('id', ''), chap.get('keywords', ''))
            cached_ctx = self._chapter_context_cache.get(ctx_key)
            if cached_ctx:
                structured_row_data = cached_ctx.get("structured_row_data", "")
                rag_data = cached_ctx.get("rag_data", "")
            else:
                structured_row_data = self.kb.get_exact_context(client, project, budget)
                rag_data = self.kb.query(client, project, chap['keywords'])
                self._cache_put(
                    self._chapter_context_cache,
                    ctx_key,
                    {"structured_row_data": structured_row_data, "rag_data": rag_data},
                    max_size=256
                )
            
            persona = PERSONAS.get(chap.get('id', 'default'), PERSONAS['default'])
            subs = "\n".join([f"- {s}" for s in chap['subs']])
            
            visual_prompt = ""
            if chap.get('visual_intent') == "gantt":
                visual_prompt = f"Mandatory Timeline Visual: [[GANTT: Jadwal Pelaksanaan | Bulan | Fase 1,0,2; Fase 2,2,4]]. Total timeline: {timeline}."
            elif chap.get('visual_intent') == "flowchart":
                visual_prompt = "Tambahkan alur tahapan metodologi dalam bentuk bullet bertingkat yang jelas (fase -> aktivitas -> output)."

            # ==========================================================
            # DYNAMIC INSTRUCTIONS: Injecting the UI Choices
            # ==========================================================
            extra = (
                f"[PROPOSAL CONTRACT]\n{proposal_contract}\n"
                "[GLOBAL] Proposal ini wajib mempertahankan kedalaman konten tingkat eksekutif dan total dokumen target 20-25 halaman. "
                "Setiap bab harus memiliki konteks spesifik klien, poin yang dapat ditindaklanjuti, dan tidak generik. Gunakan kombinasi numbering dan bullet yang rapi di setiap H2, namun tetap padat dan tidak banyak whitespace."
            )
            if chap['id'] == 'c_1':
                extra += f" [CRITICAL] Fokus pada latar belakang organisasi '{client}' dan tujuan proyek: '{project}'. Soroti driver bisnis utama: [{project_goal}]."
            elif chap['id'] == 'c_2':
                extra += f" [CRITICAL] Jabarkan kebutuhan/keinginan klien berdasarkan pain points berikut: '{notes}'. Gunakan analisis masalah yang tajam dan ringkas."
            elif chap['id'] == 'c_3':
                extra += f" [CRITICAL] Klasifikasikan kebutuhan ke Problem/Opportunity/Directive berdasarkan input: '{project_goal}'. Tetapkan jenis proyek: '{project_type}'."
            elif chap['id'] == 'c_4':
                extra += f" [CRITICAL] Gunakan framework/regulasi terpilih berikut sebagai acuan utama: '{regulations}'. Petakan langsung ke kebutuhan klien."
            elif chap['id'] == 'c_5':
                extra += f" [CRITICAL] Jelaskan alasan pemilihan metodologi untuk engagement '{service_type}' dan gunakan baseline metodologi internal: {firm_data['methodology']}."
            elif chap['id'] == 'c_6':
                extra += f" [CRITICAL] Turunkan metodologi menjadi solution design yang konkret: output, deliverable, dan target state yang dapat dieksekusi."
            elif chap['id'] == 'c_7':
                extra += f" [CRITICAL] Timeline harus sinkron dengan durasi proyek: '{timeline}'. Tampilkan aktivitas per fase, milestone, dan deliverable yang terukur."
            elif chap['id'] == 'c_8':
                extra += " [CRITICAL] Definisikan model tata kelola proyek: forum keputusan, frekuensi rapat, eskalasi isu, quality gate, dan kontrol progres."
            elif chap['id'] == 'c_9':
                extra += f" [CRITICAL] Uraikan struktur tim proyek untuk model layanan '{service_type}' dengan kapabilitas kunci, pengalaman, dan sertifikasi relevan. Referensi komposisi inti: {firm_data['team']}."
            elif chap['id'] == 'c_10':
                extra += f" [CRITICAL] Wajib menyajikan model pembiayaan dengan angka estimasi: {budget}. Sertakan termin pembayaran, model kerja, asumsi, eksklusi, dan terms komersial: {firm_data['commercial']}. Gunakan tabel markdown."
            elif chap['id'] == 'c_closing':
                contact_info = firm_profile.get('contact_info', WRITER_FIRM_NAME)
                extra += (
                    f" [CRITICAL] Ini adalah bab penutup proposal. Jangan pernah menulis label 'BAB XI' atau variasinya. "
                    f"Tunjukkan apresiasi profesional kepada klien '{client}', tegaskan komitmen kolaborasi jangka panjang, "
                    f"dan berikan langkah tindak lanjut yang jelas dan actionable. "
                    f"Wajib cantumkan informasi kontak resmi berikut secara lengkap dan akurat: {contact_info}. "
                    f"Gunakan tone hangat, profesional, dan meyakinkan."
                )

            prompt = PROPOSAL_SYSTEM_PROMPT.format(
                client=client, 
                writer_firm=WRITER_FIRM_NAME, 
                persona=persona,
                global_data=global_data, 
                client_news=client_news, 
                regulation_data=regulation_data,
                structured_row_data=structured_row_data,
                rag_data=rag_data, 
                visual_prompt=visual_prompt, 
                extra_instructions=extra,
                chapter_title=chap['title'], 
                sub_chapters=subs, 
                length_intent=chap.get('length_intent')
            )
            return {"prompt": prompt, "success": True}
        except Exception as e:
            return {"prompt": "", "success": False, "error": str(e)}

    @staticmethod
    def _contains_client_reference(content: str, client: str) -> bool:
        tokens = [
            t for t in re.findall(r"[A-Za-z0-9]{3,}", client)
            if t.lower() not in {"pt", "cv", "tbk"}
        ]
        if not tokens:
            return True
        return any(re.search(rf"\b{re.escape(tok)}\b", content, re.IGNORECASE) for tok in tokens[:3])

    def _chapter_quality_report(self, chap: Dict[str, Any], content: str, client: str) -> Dict[str, Any]:
        target_words = self._target_words(chap)
        min_words = max(280, int(target_words * 0.75))
        max_words = max(min_words + 120, int(target_words * 1.35))
        word_count = self._word_count(content)
        missing_h2 = [
            sub for sub in chap.get('subs', [])
            if not re.search(rf"(?im)^\s*##\s*{re.escape(sub)}\s*$", content)
        ]
        has_numbered_list = bool(re.search(r"(?m)^\s*\d+\.\s+\S+", content))
        has_bullet_list = bool(re.search(r"(?m)^\s*[-*]\s+\S+", content))
        has_client_ref = self._contains_client_reference(content, client)
        has_required_visual = True
        if chap.get('visual_intent') == "gantt":
            has_required_visual = "[[GANTT:" in content

        issues = []
        if missing_h2:
            issues.append("missing_h2")
        if word_count < min_words:
            issues.append("too_short")
        if word_count > max_words:
            issues.append("too_long")
        if not has_numbered_list or not has_bullet_list:
            issues.append("list_structure")
        if not has_required_visual:
            issues.append("missing_visual")
        if not has_client_ref:
            issues.append("missing_client_ref")

        return {
            "issues": issues,
            "word_count": word_count,
            "target_words": target_words,
            "min_words": min_words,
            "max_words": max_words,
            "missing_h2": missing_h2,
        }

    def _ensure_required_headings(self, chap: Dict[str, Any], content: str) -> str:
        missing_h2 = [
            sub for sub in chap.get('subs', [])
            if not re.search(rf"(?im)^\s*##\s*{re.escape(sub)}\s*$", content)
        ]
        if not missing_h2:
            return content

        patched = content.rstrip()
        for heading in missing_h2:
            patched += (
                f"\n\n## {heading}\n"
                "### Rincian Inti\n"
                "1. Aktivitas utama pada bagian ini disesuaikan langsung dengan kebutuhan bisnis klien.\n"
                "- Risiko, dependensi, dan indikator keberhasilan dijabarkan secara terukur untuk eksekusi.\n"
            )
        return patched

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _chapter_excerpt(text: str, max_words: int = 170) -> str:
        words = re.findall(r"\S+", text)
        if len(words) <= max_words:
            return " ".join(words)
        head = " ".join(words[:110])
        tail = " ".join(words[-60:])
        return f"{head} ... {tail}"

    def _apply_global_coherence(
        self,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        client: str,
        project: str
    ) -> Dict[str, str]:
        if len(chapter_outputs) < 2:
            return chapter_outputs

        snippets = []
        for chap in selected_chapters:
            content = chapter_outputs.get(chap['id'])
            if content:
                snippets.append(f"[{chap['id']}] {chap['title']} :: {self._chapter_excerpt(content)}")
        if len(snippets) < 2:
            return chapter_outputs

        snippets_text = "\n".join(snippets)
        prompt = f"""
        Audit koherensi proposal lintas bab untuk klien {client} dan inisiatif {project}.
        Ringkasan bab:
        {snippets_text}

        Keluarkan JSON murni tanpa markdown:
        {{
          "canonical_terms": [
            {{"preferred": "target state", "variants": ["kondisi target", "sasaran akhir"]}}
          ],
          "bridge_sentences": [
            {{"chapter_id": "c_2", "sentence": "Rumusan masalah ini menjadi dasar klasifikasi kebutuhan pada bab berikutnya."}}
          ]
        }}

        Aturan:
        - Maksimal 6 canonical_terms.
        - Maksimal 6 bridge_sentences.
        - sentence <= 25 kata.
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You output strictly valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'num_ctx': 16384, 'num_predict': 900, 'temperature': 0.1}
            )
            directives = self._extract_json_object(res.get('message', {}).get('content', ''))
        except Exception:
            directives = None

        if not directives:
            return chapter_outputs

        canonical_terms = directives.get("canonical_terms", [])
        bridge_sentences = directives.get("bridge_sentences", [])
        revised = dict(chapter_outputs)

        for chap_id, text in revised.items():
            updated = text
            for item in canonical_terms:
                preferred = str(item.get("preferred", "")).strip()
                variants = item.get("variants", []) or []
                if not preferred:
                    continue
                for variant in variants:
                    variant_text = str(variant).strip()
                    if not variant_text or variant_text.lower() == preferred.lower():
                        continue
                    updated = re.sub(rf"(?i)\b{re.escape(variant_text)}\b", preferred, updated)
            revised[chap_id] = updated

        for item in bridge_sentences:
            chap_id = str(item.get("chapter_id", "")).strip()
            sentence = str(item.get("sentence", "")).strip()
            if not chap_id or chap_id not in revised or not sentence:
                continue
            if sentence in revised[chap_id]:
                continue
            revised[chap_id] = revised[chap_id].rstrip() + f"\n\n- {sentence}"

        return revised

    def _generate_chapter_content(self, chap: Dict[str, Any], prompt: str, client: str) -> str:
        # One primary generation call per chapter.
        res = self.ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': (
                    f"Tulis konten untuk {chap['title']} dalam satu draft final. "
                    "Pastikan hard checks terpenuhi: H2 wajib lengkap, word range sesuai target, "
                    "ada numbered list dan bullet list, serta konten tetap konkret dan action-oriented."
                )}
            ],
            options={'num_ctx': 65536, 'num_predict': 4096, 'temperature': 0.25, 'top_p': 0.85, 'repeat_penalty': 1.1}
        )
        content = (res.get('message', {}).get('content', '') or '').strip()
        report = self._chapter_quality_report(chap, content, client)
        hard_check_keys = {"missing_h2", "too_short", "too_long", "list_structure", "missing_visual"}
        hard_issues = [i for i in report["issues"] if i in hard_check_keys]
        if not hard_issues:
            return content

        # Single retry only when hard checks fail.
        retry_prompt = (
            f"Perbaiki draft {chap['title']} agar lulus hard quality checks.\n"
            f"Issues: {', '.join(hard_issues)}\n"
            f"Word count: {report['word_count']} (target {report['target_words']}, range {report['min_words']}-{report['max_words']}).\n"
            f"Missing H2: {', '.join(report['missing_h2']) if report['missing_h2'] else '-'}\n"
            "Pertahankan fakta dan konteks. Keluarkan versi final saja.\n\n"
            f"DRAFT:\n{content}"
        )
        try:
            retry = self.ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': retry_prompt}
                ],
                options={'num_ctx': 65536, 'num_predict': 4096, 'temperature': 0.2, 'top_p': 0.85, 'repeat_penalty': 1.1}
            )
            improved = (retry.get('message', {}).get('content', '') or '').strip()
            if improved:
                content = improved
        except Exception:
            pass

        final_report = self._chapter_quality_report(chap, content, client)
        if final_report.get("missing_h2"):
            content = self._ensure_required_headings(chap, content)
            final_report = self._chapter_quality_report(chap, content, client)

        if final_report["issues"]:
            logger.warning(f"Quality checks not fully satisfied for {chap['title']}: {', '.join(final_report['issues'])}")
        return content

    def run(
        self,
        client: str,
        project: str,
        budget: str = "",
        service_type: str = "Konsultan",
        project_goal: str = "Problem",
        project_type: str = "Implementation",
        timeline: str = "TBD",
        notes: str = "",
        regulations: str = "",
        chapter_id: Optional[str] = None
    ) -> Tuple[Document, str]:
        selected_chapters = self._resolve_chapters(chapter_id)

        firm_data = self.firm_api.get_project_standards(project_type)
        firm_profile = self.firm_api.get_firm_profile()
        base_client = re.sub(r'\b(Cabang|Branch|Tbk)\b.*$|^(PT\.|CV\.)', '', client, flags=re.IGNORECASE).strip()
        research_bundle = self._get_research_bundle(base_client, regulations)
        proposal_contract = self._build_proposal_contract(
            client=client,
            project=project,
            budget=budget,
            service_type=service_type,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            selected_chapters=selected_chapters,
            research_bundle=research_bundle,
            firm_data=firm_data
        )
        logo_future = self.io_pool.submit(LogoManager.get_logo_and_color, base_client)

        context_futures = {
            chap['id']: self.io_pool.submit(
                self._fetch_chapter_context,
                chap, client, project, budget, service_type, project_goal, project_type, timeline,
                notes, regulations, firm_data, firm_profile, research_bundle, proposal_contract
            )
            for chap in selected_chapters
        }

        chapter_outputs: Dict[str, str] = {}
        for chap in selected_chapters:
            ctx = context_futures[chap['id']].result()
            if not ctx['success']:
                continue
            try:
                chapter_outputs[chap['id']] = self._generate_chapter_content(chap, ctx['prompt'], client)
            except Exception as e:
                logger.error(f"Generation Error for {chap['title']}: {e}")

        chapter_outputs = self._apply_global_coherence(chapter_outputs, selected_chapters, client, project)

        try:
            logo_stream, theme_color = logo_future.result(timeout=8)
        except Exception:
            logo_stream, theme_color = None, DEFAULT_COLOR

        doc = Document()
        StyleEngine.apply_document_styles(doc)
        
        # Personalized Cover Generation
        for _ in range(2):
            doc.add_paragraph()

        if logo_stream:
            try:
                logo_stream.seek(0)
                cover_logo = doc.add_paragraph()
                cover_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cover_logo.add_run().add_picture(logo_stream, width=Inches(1.8))
            except (UnrecognizedImageError, OSError, ValueError) as e:
                logger.warning(f"Logo skipped due to unsupported image format: {e}")

        title = doc.add_paragraph("ARSITEKTUR PROPOSAL")
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.runs[0]
        title_run.bold = True
        title_run.font.size = Pt(30)
        title_run.font.color.rgb = RGBColor(*theme_color)

        if len(selected_chapters) == 1:
            subtitle_text = f"{selected_chapters[0]['title']} ({service_type} – {project_type})"
        else:
            subtitle_text = f"{service_type} – {project_type}"

        subtitle = doc.add_paragraph(subtitle_text)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.runs[0].font.size = Pt(14)

        doc.add_paragraph()
        client_line = doc.add_paragraph(f"Untuk: {client}")
        client_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
        client_line.runs[0].bold = True
        client_line.runs[0].font.size = Pt(16)

        project_line = doc.add_paragraph(f"Inisiatif: {project}")
        project_line.alignment = WD_ALIGN_PARAGRAPH.CENTER

        meta = doc.add_paragraph(
            f"Durasi: {timeline} | Estimasi Investasi: {budget or 'Menyesuaikan ruang lingkup'} | Tanggal: {datetime.now().strftime('%d %B %Y')}"
        )
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

        firm_info = firm_profile.get('contact_info', WRITER_FIRM_NAME)
        contact = doc.add_paragraph(f"Disusun oleh {WRITER_FIRM_NAME}\n{firm_info}")
        contact.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_page_break()

        rendered_any = False
        for i, chap in enumerate(selected_chapters):
            content = chapter_outputs.get(chap['id'], '').strip()
            if not content:
                continue
            rendered_any = True
            h = doc.add_heading(chap['title'], level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            h.runs[0].font.color.rgb = RGBColor(*theme_color)
            DocumentBuilder.process_content(doc, content, theme_color, chap['title'])

            has_next = any(
                chapter_outputs.get(next_chap['id'], '').strip()
                for next_chap in selected_chapters[i + 1:]
            )
            if has_next:
                doc.add_page_break()

        if not rendered_any:
            doc.add_paragraph("Konten proposal belum berhasil digenerate. Mohon ulangi proses.")

        base_name = f"Proposal_{client}_{project}"
        if len(selected_chapters) == 1:
            chapter_slug = re.sub(r'[^A-Za-z0-9]+', '_', selected_chapters[0]['title']).strip('_')
            base_name = f"{base_name}_{chapter_slug}"

        return doc, base_name.replace(" ", "_")
