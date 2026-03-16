"""
Core business logic for the Proposal Generation Engine.
Handles API integrations, Data retrieval, OSINT gathering (Serper.dev), and Document composition.
"""

import os
import io
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import concurrent.futures

import requests
import pandas as pd
from bs4 import BeautifulSoup
import markdown

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from PIL import Image, ImageStat

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, exc as sa_exc

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from ollama import Client 

from config import (
    SERPER_API_KEY, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, DB_URI,
    WRITER_FIRM_NAME, DEFAULT_COLOR, UNIVERSAL_STRUCTURE,
    PROPOSAL_SYSTEM_PROMPT, DATA_MAPPING, DEMO_MODE, FIRM_API_URL, 
    API_AUTH_TOKEN, MOCK_FIRM_STANDARDS, MOCK_FIRM_PROFILE
)

logger = logging.getLogger(__name__)

# =====================================================================
# FIRM API ADAPTER
# =====================================================================
class FirmAPIClient:
    """Handles communication with the internal firm database/API."""
    
    def __init__(self) -> None:
        self.demo_mode: bool = DEMO_MODE
        self.base_url: str = FIRM_API_URL
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {API_AUTH_TOKEN}"}

    def get_project_standards(self, project_type: str) -> Dict[str, str]:
        if self.demo_mode:
            return MOCK_FIRM_STANDARDS.get(project_type, MOCK_FIRM_STANDARDS.get("Implementation", {}))
            
        try:
            response = requests.get(f"{self.base_url}/standards/{project_type}", headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logger.error(f"Internal API Connection failed: {error}")
            return {"methodology": "TBD", "team": "TBD", "commercial": "TBD"}

    def get_firm_profile(self) -> Dict[str, str]:
        if self.demo_mode:
            return MOCK_FIRM_PROFILE
            
        try:
            response = requests.get(f"{self.base_url}/firm-profile", headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logger.error(f"Internal API Profile fetch failed: {error}")
            return {"contact_info": "Kantor Pusat Terdaftar", "portfolio_highlights": "Penyedia Solusi IT Terkemuka"}


# =====================================================================
# KNOWLEDGE BASE (Vector DB via Chroma)
# =====================================================================
class KnowledgeBase:
    """Manages local project history vectors for RAG implementations."""
    
    def __init__(self, database_uri: str) -> None:
        self.engine = create_engine(database_uri)
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings", model_name=EMBED_MODEL
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="projects_db", embedding_function=self.embedding_fn
        )
        self.dataframe: Optional[pd.DataFrame] = None
        self.refresh_data()

    def refresh_data(self) -> bool:
        """Syncs SQL/CSV data into ChromaDB vector space."""
        try:
            self.dataframe = pd.read_sql("SELECT * FROM projects", self.engine)
        except sa_exc.SQLAlchemyError:
            if not os.path.exists("db.csv"):
                return False
            
            try:
                raw_df = pd.read_csv("db.csv")
                raw_df.columns = [col.strip() for col in raw_df.columns]
                rename_dict = {val: key for key, val in DATA_MAPPING.items()}
                raw_df.rename(columns=rename_dict, inplace=True)
                raw_df.to_sql("projects", self.engine, index=False, if_exists='replace')
                self.dataframe = raw_df
            except Exception as error:
                logger.error(f"Data mapping error: {error}")
                return False
                
        try:
            existing_ids = set(self.collection.get()['ids'])
            new_ids_map = {str(idx): row for idx, row in self.dataframe.iterrows()}
            new_ids_set = set(new_ids_map.keys())
            
            ids_to_delete = list(existing_ids - new_ids_set)
            ids_to_add = list(new_ids_set - existing_ids)
            
            if ids_to_delete: 
                self.collection.delete(ids_to_delete)
                
            if ids_to_add:
                batch_size = 500 
                for i in range(0, len(ids_to_add), batch_size):
                    batch_ids = ids_to_add[i:i + batch_size]
                    docs, metas = [], []
                    for b_id in batch_ids:
                        row = new_ids_map[b_id]
                        text_rep = " | ".join([f"{col}: {val}" for col, val in row.items()])
                        docs.append(text_rep)
                        metas.append(row.astype(str).to_dict())
                    self.collection.add(documents=docs, metadatas=metas, ids=batch_ids)
            return True
        except Exception as error:
            logger.error(f"Vector DB sync failed: {error}")
            return False


# =====================================================================
# UNRESTRICTED OSINT RESEARCHER (Serper.dev)
# =====================================================================
class Researcher:
    """Executes live, highly accurate Open Source Intelligence gathering."""
    
    @staticmethod
    def get_system_geolocation() -> str:
        try:
            loc_data = requests.get('https://ipinfo.io/json', timeout=2).json()
            return loc_data.get('country', 'ID').lower()
        except requests.RequestException:
            return 'id' 

    @staticmethod
    @lru_cache(maxsize=256)
    def search(query: str, limit: int = 5, up_to_the_second: bool = False, use_news_endpoint: bool = False) -> Optional[Dict]:
        if "YOUR_SERPER" in SERPER_API_KEY: 
            return None
            
        endpoint = "news" if use_news_endpoint else "search"
        url = f"https://google.serper.dev/{endpoint}"
        
        payload_dict = {
            "q": query,
            "num": limit,
            "gl": Researcher.get_system_geolocation(),
            "autocorrect": True
        }
        
        if up_to_the_second:
            payload_dict["tbs"] = "sbd:1"
            
        try:
            response = requests.post(
                url, 
                headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}, 
                data=json.dumps(payload_dict), 
                timeout=6
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logger.warning(f"Serper API Error on query '{query}': {error}")
            return None

    @staticmethod
    @lru_cache(maxsize=128)
    def get_entity_profile(entity_name: str) -> str:
        data = Researcher.search(f'"{entity_name}" profil perusahaan OR "tentang kami"', limit=3)
        if not data or 'organic' not in data: return f"{entity_name}"
        return "\n".join([item.get('snippet', '') for item in data['organic']])

    @staticmethod
    @lru_cache(maxsize=128)
    def get_financial_data(entity_name: str) -> str:
        query = f'"{entity_name}" (laporan keuangan OR pendapatan OR revenue OR pendanaan OR laba bersih) 2025 OR 2026'
        data = Researcher.search(query, limit=4)
        if not data or 'organic' not in data: 
            return "Data finansial spesifik tidak terpublikasi."
        return "\n".join([item.get('snippet', '') for item in data['organic']])

    @staticmethod
    @lru_cache(maxsize=128)
    def get_latest_client_news(client_name: str) -> str:
        data = Researcher.search(f'"{client_name}" inovasi OR investasi', limit=4, up_to_the_second=True, use_news_endpoint=True)
        if not data or 'news' not in data: return "Tidak ada berita terbaru."
        return "\n".join([item.get('snippet', '') for item in data['news']])

    @staticmethod
    @lru_cache(maxsize=128)
    def get_regulatory_data(regulation_name: str) -> str:
        if not regulation_name: return ""
        data = Researcher.search(f'Ringkasan kepatuhan mandat {regulation_name}', limit=3)
        if not data or 'organic' not in data: return ""
        return "\n".join([item.get('snippet', '') for item in data['organic']])

    @staticmethod
    @lru_cache(maxsize=128)
    def get_collaboration_data(client: str, firm: str) -> str:
        data = Researcher.search(f'"{client}" "{firm}" kerjasama OR proyek', limit=3)
        if not data or 'organic' not in data: return ""
        return "\n".join([item.get('snippet', '') for item in data['organic']])


class LogoManager:
    """Manages automatic corporate logo fetching and palette extraction."""
    @staticmethod
    def get_logo_and_color(client_name: str) -> Tuple[Optional[io.BytesIO], Tuple[int, int, int]]:
        if "YOUR_SERPER" in SERPER_API_KEY: 
            return None, DEFAULT_COLOR
            
        payload = json.dumps({
            "q": f"{client_name} company corporate logo png transparent",
            "num": 2,
            "gl": Researcher.get_system_geolocation()
        })
        try:
            res = requests.post(
                "https://google.serper.dev/images", 
                headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}, 
                data=payload, timeout=6
            ).json()
            
            if 'images' in res:
                for item in res['images']:
                    img_url = item.get('imageUrl')
                    if not img_url: continue
                    
                    img_resp = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                    if img_resp.status_code == 200:
                        stream = io.BytesIO(img_resp.content)
                        img = Image.open(stream).convert('RGB')
                        img.thumbnail((150, 150))
                        
                        dom_color = list(map(int, ImageStat.Stat(img).mean[:3]))
                        luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                        if luminance > 130:  
                            darken_factor = 130 / luminance
                            dom_color = [max(0, min(255, int(c * darken_factor))) for c in dom_color]
                            
                        stream.seek(0)
                        return stream, tuple(dom_color)
        except Exception:
            pass
            
        return None, DEFAULT_COLOR


# =====================================================================
# DOCUMENT RENDERING & FORMATTING ENGINE
# =====================================================================
class StyleEngine:
    """Applies strict Microsoft Word formatting templates."""
    @staticmethod
    def apply_document_styles(doc: Document) -> None:
        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.space_after = Pt(8) 
        
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)

class ChartEngine:
    @staticmethod
    def _get_plt_color(theme_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(c/255.0 for c in theme_color)

    @staticmethod
    def create_bar_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = data_str.split('|')
            title_str, ylabel_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip() if len(parts) == 3 else ("Data Analysis", "Value", data_str)
            
            labels, values = [], []
            for pair in raw_data.split(';'):
                if ',' in pair:
                    l, v = pair.split(',', 1)
                    labels.append(l.strip())
                    values.append(float(re.sub(r'[^\d.]', '', v)))
                    
            if not labels: return None
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(labels, values, color=ChartEngine._get_plt_color(theme_color), alpha=0.9, width=0.5, edgecolor='white')
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=20)
            ax.set_ylabel(ylabel_str, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=20, ha='right', fontsize=9)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

class DocumentBuilder:
    """Parses sanitized HTML/Markdown from the LLM into formatted Docx blocks."""
    
    @staticmethod
    def parse_html_to_docx(doc: Document, html_content: str, theme_color: Tuple[int, int, int]) -> None:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup.children:
            if element.name is None: 
                continue
            
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                level = int(element.name[1])
                heading = doc.add_heading(element.get_text().strip(), level=level)
                heading.paragraph_format.space_before = Pt(20 if level == 2 else 14)
                for run in heading.runs:
                    run.font.color.rgb = RGBColor(*theme_color)
                    run.font.name = 'Arial'
                    run.font.size = Pt(14 if level == 2 else 12)
                    run.bold = True
                    
            elif element.name == 'p':
                text = element.get_text().strip()
                if not text: continue
                p = doc.add_paragraph()
                DocumentBuilder._process_inline_html(p, element)
                
            elif element.name in ['ul', 'ol']:
                style = 'List Bullet' if element.name == 'ul' else 'List Number'
                for li in element.find_all('li'):
                    p = doc.add_paragraph(style=style)
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(4)
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
                            cell.text = col.get_text(separator='\n').strip()
                            if row.find('th') or i == 0:
                                p = cell.paragraphs[0]
                                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                for run in p.runs: run.bold = True
                                tcPr = cell._element.get_or_add_tcPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:val'), 'clear')
                                shd.set(qn('w:color'), 'auto')
                                shd.set(qn('w:fill'), 'E5E7EB')
                                tcPr.append(shd)

    @staticmethod
    def _process_inline_html(paragraph, element) -> None:
        for child in element.children:
            if child.name in ['strong', 'b']: paragraph.add_run(child.get_text()).bold = True
            elif child.name in ['em', 'i']: paragraph.add_run(child.get_text()).italic = True
            elif child.name is None: paragraph.add_run(str(child))
            else: DocumentBuilder._process_inline_html(paragraph, child)

    @staticmethod
    def process_content(doc: Document, raw_text: str, theme_color: Tuple[int, int, int], chapter_title: str = "") -> None:
        raw_text = re.sub(r'^[ \t]*[\*\•\➢\+][ \t]+', '- ', raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r'([^\n])\n(- |1\. )', r'\1\n\n\2', raw_text)

        clean_lines = []
        for line in raw_text.split('\n'):
            stripped = line.strip()
            
            if stripped.startswith('[[CHART:') and stripped.endswith(']]'):
                data = stripped.replace('[[CHART:', '').replace(']]', '').strip()
                img = ChartEngine.create_bar_chart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(5.5))
                continue
                
            if stripped.startswith('#') and chapter_title.lower() in stripped.lower():
                continue 
                
            clean_lines.append(line) 
            
        md_text = "\n".join(clean_lines)
        html = markdown.markdown(md_text, extensions=['tables'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)

    @staticmethod
    def create_cover(doc: Document, client: str, project_type: str, logo_stream: Optional[io.BytesIO] = None, theme_color: Tuple[int, int, int] = DEFAULT_COLOR) -> None:
        StyleEngine.apply_document_styles(doc)
        for _ in range(3): doc.add_paragraph()
        
        if logo_stream:
            try:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(logo_stream, width=Inches(3))
            except Exception as error:
                logger.warning(f"Cover logo insertion failed: {error}")
                
        doc.add_paragraph()
        t = doc.add_paragraph("PROPOSAL STRATEGIS")
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        t.runs[0].font.size = Pt(18)
        
        c = doc.add_paragraph(client.upper())
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = c.runs[0]
        run.bold = True
        run.font.size = Pt(28)
        run.font.color.rgb = RGBColor(*theme_color)
        
        p_name = doc.add_paragraph(f"Inisiatif: {project_type}")
        p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_name.runs[0].font.size = Pt(14)
        p_name.runs[0].italic = True
        
        for _ in range(5): doc.add_paragraph()
        s = doc.add_paragraph(f"Disusun Oleh:\n{WRITER_FIRM_NAME}")
        s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()


# =====================================================================
# CORE GENERATOR PIPELINE
# =====================================================================
class ProposalGenerator:
    """Orchestrates the asynchronous data fetching and sequential LLM execution."""
    
    def __init__(self, kb_instance: KnowledgeBase) -> None:
        self.ollama_client = Client(host=OLLAMA_HOST)
        self.kb = kb_instance
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.firm_api = FirmAPIClient()

    def suggest_budget(self, client_name: str) -> str:
        """Exposed method for the frontend UI to suggest a budget dynamically."""
        clean_regex = r'\b(Cabang|Branch|Region|Area|Tbk)\b.*$|^(PT\.|PT\s+|CV\.|CV\s+)'
        base_client = re.sub(clean_regex, '', client_name, flags=re.IGNORECASE).strip()
        
        financial_data = Researcher.get_financial_data(base_client)
        
        if "tidak terpublikasi" in financial_data.lower():
            return "Data finansial publik tidak ditemukan. Estimasi berbasis skala industri standar."
            
        prompt = f"""
        Tugas: Berikan saran budget IT.
        Klien: {base_client}
        Konteks Finansial OSINT: {financial_data}
        
        Instruksi: Berdasarkan kekuatan finansial tersebut, berikan 1 kalimat singkat (maks 15 kata) yang merekomendasikan target budget (dalam Rupiah) untuk inisiatif IT enterprise.
        Contoh: "Berdasarkan Q3 Revenue 2T, budget rasional adalah Rp 1M - Rp 3M."
        """
        
        try:
            res = self.ollama_client.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 60}
            )
            return res['message']['content'].strip()
        except Exception as e:
            logger.error(f"Budget suggestion inference failed: {e}")
            return "Gagal menghasilkan saran cerdas."

    def _fetch_chapter_context(self, chap: Dict, client: str, project_status: str, project_type: str, scope: str, project_goal: str, timeline: str, budget: str, regulations: str, notes: str, firm_data: Dict, firm_profile: Dict, futures: Dict) -> Dict[str, Any]:
        """Assembles the specific context window for a single chapter."""
        try:
            global_data = futures['profile'].result(timeout=6) if futures.get('profile') else ""
            client_news = futures['news'].result(timeout=6) if futures.get('news') else ""
            client_financials = futures['financial'].result(timeout=6) if futures.get('financial') else "Data finansial tidak tersedia."
            regulation_data = futures['regulations'].result(timeout=6) if futures.get('regulations') else ""
            collab_data = futures['collab'].result(timeout=6) if futures.get('collab') else ""

            subs = "\n".join([f"- {s}" for s in chap['subs']])

            extra_instructions = ""
            if chap['id'] == 'c_1':
                extra_instructions = f"[MANDATORY] Highlight the urgency based on their status: {global_data}"
            elif chap['id'] == 'c_4':
                extra_instructions = f"[MANDATORY] Detail how the solution addresses these mandates: {regulations}. Reference context: {regulation_data}"
            elif chap['id'] == 'c_8':
                extra_instructions = f"[MANDATORY] Map the team structure specifically to: {firm_data['team']}"
            elif chap['id'] == 'c_9':
                extra_instructions = f"[MANDATORY] Analisa metrik finansial berikut: {client_financials}. Buatlah 3 opsi harga berjenjang (Esensial, Rekomendasi, Premium) yang secara logis sepadan dengan skala bisnis tersebut, di mana Opsi Rekomendasi adalah {budget}."
            elif chap['id'] == 'c_10':
                extra_instructions = f"[MANDATORY] End forcefully. Contact info: {firm_profile.get('contact_info')}"

            prompt = PROPOSAL_SYSTEM_PROMPT.format(
                client=client,
                writer_firm=WRITER_FIRM_NAME,
                global_data=global_data,
                client_news=client_news,
                client_financials=client_financials,
                regulation_data=regulation_data,
                collab_data=collab_data,
                firm_api_portfolio=firm_profile.get("portfolio_highlights", ""),
                firm_api_contact=firm_profile.get("contact_info", ""),
                project_status=project_status,
                project_type=project_type,
                scope=scope,
                outcome=project_goal,
                timeline=timeline,
                budget=budget,
                notes=notes,
                extra_instructions=extra_instructions,
                chapter_title=chap['title'],
                sub_chapters=subs,
                length_intent=chap.get('length_intent', '')
            )

            return {"prompt": prompt, "success": True}
        except Exception as error:
            logger.error(f"Context assembly failed for {chap['title']}: {error}")
            return {"prompt": "", "success": False, "error": str(error)}

    def run(self, client: str, project_status: str, project_type: str, scope: str, project_goal: str, timeline: str, budget: str, regulations: str, notes: str) -> Tuple[Document, str]:
        """Main execution loop for document generation."""
        logger.info(f"Initiating Generation Pipeline for: {client}")
        
        firm_data = self.firm_api.get_project_standards(project_type)
        firm_profile = self.firm_api.get_firm_profile()
        
        clean_regex = r'\b(Cabang|Branch|Region|Area|Tbk)\b.*$|^(PT\.|PT\s+|CV\.|CV\s+)'
        base_client = re.sub(clean_regex, '', client, flags=re.IGNORECASE).strip()
        base_firm = re.sub(clean_regex, '', WRITER_FIRM_NAME, flags=re.IGNORECASE).strip()

        research_futures = {
            'profile': self.io_pool.submit(Researcher.get_entity_profile, base_client),
            'news': self.io_pool.submit(Researcher.get_latest_client_news, base_client),
            'financial': self.io_pool.submit(Researcher.get_financial_data, base_client),
            'regulations': self.io_pool.submit(Researcher.get_regulatory_data, regulations),
            'collab': self.io_pool.submit(Researcher.get_collaboration_data, base_client, base_firm)
        }
        logo_future = self.io_pool.submit(LogoManager.get_logo_and_color, base_client) 

        context_futures = {}
        for chap in UNIVERSAL_STRUCTURE:
            context_futures[chap['id']] = self.io_pool.submit(
                self._fetch_chapter_context, chap, client, project_status, project_type, scope, project_goal, timeline, budget, regulations, notes, firm_data, firm_profile, research_futures
            )

        try:
            logo_stream, theme_color = logo_future.result(timeout=10)
        except Exception:
            logo_stream, theme_color = None, DEFAULT_COLOR

        doc = Document()
        DocumentBuilder.create_cover(doc, client, project_type, logo_stream, theme_color)
        
        for idx, chap in enumerate(UNIVERSAL_STRUCTURE):
            ctx_result = context_futures[chap['id']].result()
            if ctx_result.get('success'):
                try:
                    response = self.ollama_client.chat(
                        model=LLM_MODEL, 
                        messages=[
                            {'role': 'system', 'content': ctx_result['prompt']}, 
                            {'role': 'user', 'content': f"Tulis {chap['title']} tanpa basa-basi."}
                        ],
                        options={'num_ctx': 4096, 'num_predict': 1024} 
                    )
                    
                    heading = doc.add_heading(chap['title'], level=1)
                    heading.runs[0].font.color.rgb = RGBColor(*theme_color)
                    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    DocumentBuilder.process_content(doc, response['message']['content'], theme_color, chap['title'])
                    
                    if idx < len(UNIVERSAL_STRUCTURE) - 1: 
                        doc.add_page_break()
                except Exception as error: 
                    logger.error(f"LLM Generation failure on {chap['title']}: {error}")

        safe_filename = f"Proposal_{base_client.replace(' ', '_')}_{project_type}"
        return doc, safe_filename