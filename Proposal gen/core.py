import os
import io
import re
import logging
import requests
import pandas as pd
import chromadb
from chromadb.config import Settings
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from PIL import Image, ImageStat
from difflib import SequenceMatcher

# SQL ORM untuk migrasi Database
from sqlalchemy import create_engine

# Robust Parsing
import markdown
from bs4 import BeautifulSoup

# Docx
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# AI Clients
from ollama import Client 
from chromadb.utils import embedding_functions

# Import dari config
from config import (
    GOOGLE_API_KEY, GOOGLE_CX_ID, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, DB_URI,
    WRITER_FIRM_NAME, DEFAULT_COLOR, TRAINING_STRUCTURE, CONSULTING_STRUCTURE, 
    PERSONAS, PROPOSAL_SYSTEM_PROMPT, DATA_MAPPING, 
    DEMO_MODE, FIRM_API_URL, API_AUTH_TOKEN, MOCK_FIRM_STANDARDS
)

logger = logging.getLogger(__name__)

# =====================================================================
# FIRM API ADAPTER (Siap untuk Handover)
# =====================================================================
class FirmAPIClient:
    """Mengambil standar Hak Kekayaan Intelektual perusahaan (Methodology, Team, Pricing)"""
    def __init__(self):
        self.demo_mode = DEMO_MODE
        self.base_url = FIRM_API_URL
        self.headers = {"Authorization": f"Bearer {API_AUTH_TOKEN}"}

    def get_project_standards(self, project_type):
        """Menarik Metodologi, Tim, dan Harga berdasarkan tipe proyek"""
        if self.demo_mode:
            logger.info(f"[DEMO MODE] Menggunakan Mock Data Internal untuk tipe: {project_type}")
            return MOCK_FIRM_STANDARDS.get(project_type, MOCK_FIRM_STANDARDS.get("Implementation"))
        else:
            logger.info(f"[PROD MODE] Mengambil standar dari API Perusahaan: {self.base_url}")
            try:
                response = requests.get(f"{self.base_url}/standards/{project_type}", headers=self.headers, timeout=5)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Gagal terhubung ke API Internal: {e}")
                return {"methodology": "TBD", "team": "TBD", "commercial": "TBD"}

# =====================================================================
# KNOWLEDGE BASE & OSINT (RESEARCHER)
# =====================================================================
class KnowledgeBase:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        self.chroma = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings", model_name=EMBED_MODEL
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", embedding_function=self.embed_fn
        )
        self.df = None
        self.refresh_data()

    def refresh_data(self):
        try:
            self.df = pd.read_sql("SELECT * FROM projects", self.engine)
        except Exception:
            if os.path.exists("db.csv"):
                raw_df = pd.read_csv("db.csv")
                raw_df.columns = [c.strip() for c in raw_df.columns]
                rename_dict = {v: k for k, v in DATA_MAPPING.items()}
                raw_df.rename(columns=rename_dict, inplace=True)
                
                def format_rupiah_range(val):
                    try:
                        clean_str = re.sub(r'[^\d]', '', str(val))
                        if not clean_str: return val
                        base_val = int(clean_str)
                        ceiling_val = base_val + 5000000
                        floor_str = f"Rp. {base_val:,}".replace(',', '.')
                        ceiling_str = f"Rp. {ceiling_val:,}".replace(',', '.')
                        return f"{floor_str} - {ceiling_str}"
                    except Exception: return val

                if 'budget' in raw_df.columns:
                    raw_df['budget'] = raw_df['budget'].apply(format_rupiah_range)
                    
                raw_df.to_sql("projects", self.engine, index=False, if_exists='replace')
                self.df = raw_df
            else:
                return False
            
        existing = self.collection.get()['ids']
        if existing: self.collection.delete(existing)
        
        ids, docs, metas = [], [], []
        for idx, row in self.df.iterrows():
            client = row.get('entity', f'Client_{idx}') 
            project = row.get('topic', f'Project_{idx}')
            text_rep = " | ".join([f"{col}: {val}" for col, val in row.items()])
            ids.append(str(idx))
            docs.append(text_rep)
            metas.append(row.astype(str).to_dict())
            
        if ids: self.collection.add(documents=docs, metadatas=metas, ids=ids)
        return True

    def get_exact_context(self, entity, topic, budget=None):
        if self.df is None or self.df.empty: return "No data."
        try:
            match = self.df[(self.df['entity'] == entity) & (self.df['topic'] == topic)]
            if budget and not match.empty: match = match[match['budget'] == budget]
            if not match.empty:
                return "".join([f"- {k.capitalize()}: {v}\n" for k, v in match.iloc[0].to_dict().items()])
            return "No data."
        except Exception: return ""

    def query(self, client, project, context_keywords=None):
        try:
            res = self.collection.query(query_texts=[f"{project} for {client} {context_keywords or ''}"], n_results=2)
            if res['documents'] and len(res['documents'][0]) > 0: return "\n".join(res['documents'][0])
        except Exception: return ""

class Researcher:
    @staticmethod
    def search(query, limit=2):
        if "YOUR_GOOGLE" in GOOGLE_API_KEY: return None
        try:
            params = {'q': query, 'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX_ID, 'num': limit, 'gl': 'id'}
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception: return None

    @staticmethod
    def fetch_page_content(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.content, 'html.parser')
            texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            return ' '.join([t.get_text(strip=True) for t in texts])[:4000]
        except Exception: return ""

    @staticmethod
    def get_entity_profile(entity_name):
        res = Researcher.search(f"{entity_name} profil perusahaan indonesia", limit=1)
        if not res or 'items' not in res: return f"{entity_name}"
        return res['items'][0].get('snippet', '')
        
    @staticmethod
    def get_contact_details(entity_name):
        query = f"{entity_name} alamat nomor telepon email contact details"
        res = Researcher.search(query, limit=2)
        if not res or 'items' not in res: return ""
        return "\n".join([i.get('snippet', '') for i in res['items']])

    @staticmethod
    def get_collaboration_data(client, firm):
        res = Researcher.search(f"\"{client}\" \"{firm}\" partnership OR project", limit=2)
        if not res or 'items' not in res: return "Tidak ada rekam jejak kolaborasi publik."
        return "\n".join([i.get('snippet', '') for i in res['items']])

    @staticmethod
    def get_latest_client_news(client_name):
        res = Researcher.search(f"{client_name} berita teknologi masalah transformasi 2026", limit=2)
        if not res or 'items' not in res: return "Tidak ada berita relevan terbaru."
        return "\n".join([i.get('snippet', '') for i in res['items']])

    @staticmethod
    def get_regulatory_data(regulation_name):
        if not regulation_name: return "Tidak ada regulasi spesifik."
        res = Researcher.search(f"Ringkasan kepatuhan mandat {regulation_name}", limit=2)
        if not res or 'items' not in res: return f"Merujuk pada standar umum {regulation_name}."
        return "\n".join([i.get('snippet', '') for i in res['items']])

class LogoManager:
    @staticmethod
    def get_logo_and_color(client_name):
        if "YOUR_GOOGLE" in GOOGLE_API_KEY: return None, DEFAULT_COLOR
        try:
            params = {
                'q': f"{client_name} company corporate logo png transparent", 
                'key': GOOGLE_API_KEY, 
                'cx': GOOGLE_CX_ID, 
                'num': 3, 
                'searchType': 'image'
            }
            res = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=5).json()
            if 'items' in res:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
                for item in res['items']:
                    try:
                        img_resp = requests.get(item['link'], headers=headers, timeout=5)
                        if img_resp.status_code == 200:
                            stream = io.BytesIO(img_resp.content)
                            img = Image.open(stream).convert('RGB')
                            img.thumbnail((150, 150))
                            dom_color = list(map(int, ImageStat.Stat(img).mean[:3]))
                            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                            if luminance > 120:  
                                darken_factor = 120 / luminance
                                dom_color = [max(0, min(255, int(c * darken_factor))) for c in dom_color]
                            stream.seek(0)
                            return stream, tuple(dom_color)
                    except Exception: continue
        except Exception: pass
        return None, DEFAULT_COLOR

class StyleEngine:
    @staticmethod
    def apply_document_styles(doc):
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
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
    def _get_plt_color(theme_color):
        return tuple(c/255 for c in theme_color)

    @staticmethod
    def create_bar_chart(data_str, theme_color):
        try:
            parts = data_str.split('|')
            if len(parts) == 3: title_str, ylabel_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else: title_str, ylabel_str, raw_data = "Data Analysis", "Value", data_str
            labels, values = [], []
            for p in raw_data.split(';'):
                if ',' in p:
                    l, v = p.split(',', 1)
                    labels.append(l.strip())
                    values.append(float(re.sub(r'[^\d.]', '', v)))
            if not labels: return None
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bars = ax.bar(labels, values, color=ChartEngine._get_plt_color(theme_color), alpha=0.9, width=0.5, zorder=3, edgecolor='white', linewidth=1)
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=20, color='#222222')
            ax.set_ylabel(ylabel_str, fontsize=10, color='#444444', fontweight='bold')
            ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=20, ha='right', fontsize=9)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values)*0.02), f'{yval:g}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
            ax.set_ylim(0, max(values) + (max(values) * 0.15))
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception: return None

    @staticmethod
    def create_gantt_chart(data_str, theme_color):
        try:
            parts = data_str.split('|')
            if len(parts) == 3: title_str, unit_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else: title_str, unit_str, raw_data = "Implementation Timeline", "Duration", data_str
            tasks = []
            for p in raw_data.split(';'):
                t_parts = p.split(',')
                if len(t_parts) >= 3:
                    tasks.append({"task": t_parts[0].strip(), "start": float(re.sub(r'[^\d.]', '', t_parts[1])), "dur": float(re.sub(r'[^\d.]', '', t_parts[2]))})
            if not tasks: return None
            tasks = tasks[::-1] 
            names = [t['task'] for t in tasks]
            starts = [t['start'] for t in tasks]
            durs = [t['dur'] for t in tasks]
            fig, ax = plt.subplots(figsize=(8.5, max(4, len(tasks)*0.8)))
            for i, (name, start, dur) in enumerate(zip(names, starts, durs)):
                rect = patches.FancyBboxPatch((start, i-0.3), dur, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1", ec="#ffffff", fc=ChartEngine._get_plt_color(theme_color), alpha=0.9, lw=1.5)
                ax.add_patch(rect)
                ax.text(start + (dur / 2), i, f"{dur:g} {unit_str}", ha='center', va='center', color='white', fontweight='bold', fontsize=9, zorder=5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10, fontweight='medium')
            ax.set_xlabel(f"Timeline ({unit_str})", fontsize=10, fontweight='bold', labelpad=10)
            ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20, color='#222222')
            ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
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
        except Exception: return None

    @staticmethod
    def create_flowchart(data_str, theme_color):
        try:
            steps = [s.strip() for s in data_str.split('->')]
            if len(steps) < 2: return None
            steps = ["\n".join(textwrap.wrap(s, width=18)) for s in steps]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis('off')
            n_steps = len(steps)
            x_pos = [i * 2.5 for i in range(n_steps)]
            y_pos = 0.5
            for i in range(n_steps - 1):
                ax.annotate("", xy=(x_pos[i+1]-1.0, y_pos), xytext=(x_pos[i]+1.0, y_pos), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5, mutation_scale=15))
            for i, step in enumerate(steps):
                box = patches.FancyBboxPatch((x_pos[i]-1.0, y_pos-0.4), 2.0, 0.8, boxstyle="round,pad=0.1,rounding_size=0.2", fc=ChartEngine._get_plt_color(theme_color), ec="#2c3e50", alpha=0.9, zorder=2)
                ax.add_patch(box)
                ax.text(x_pos[i], y_pos, step, ha="center", va="center", size=9, color="white", fontweight='bold', zorder=3)
            ax.set_xlim(-1.2, (n_steps-1)*2.5 + 1.2)
            ax.set_ylim(0, 1)
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=200, transparent=True)
            plt.close()
            img.seek(0)
            return img
        except Exception: return None

    @staticmethod
    def create_math_image(latex_str, theme_color):
        try:
            fig = plt.figure(figsize=(6, 0.8))
            clean_tex = latex_str.replace('*', r'\times').replace('=', r'\=').strip()
            plt.text(0.5, 0.5, f"${clean_tex}$", fontsize=16, ha='center', va='center', color='#222222', fontweight='normal')
            plt.axis('off')
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=200, transparent=True)
            plt.close()
            img.seek(0)
            return img
        except Exception: return None

class DocumentBuilder:
    @staticmethod
    def _remove_duplicate_header(raw_text, title):
        lines = raw_text.strip().split('\n')
        if not lines: return raw_text
        first_line = lines[0].strip().replace('#', '').strip()
        ratio = SequenceMatcher(None, first_line.lower(), title.lower()).ratio()
        if ratio > 0.6 or first_line.lower() in title.lower(): return "\n".join(lines[1:]).strip()
        return raw_text

    @staticmethod
    def parse_html_to_docx(doc, html_content, theme_color):
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.children:
            if element.name is None: continue
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                level = int(element.name[1])
                p = doc.add_heading(element.get_text().strip(), level=level)
                p.paragraph_format.space_before = Pt(24 if level == 2 else 16)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.keep_with_next = True 
                for run in p.runs:
                    run.font.color.rgb = RGBColor(*theme_color)
                    run.font.name = 'Arial'
                    run.font.size = Pt(14 if level == 2 else 12)
                    run.bold = True
            elif element.name == 'p':
                text = element.get_text().strip()
                if not text: continue
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                DocumentBuilder._process_inline_html(p, element)
            elif element.name in ['ul', 'ol']:
                style = 'List Bullet' if element.name == 'ul' else 'List Number'
                for li in element.find_all('li'):
                    p = doc.add_paragraph(style=style)
                    DocumentBuilder._process_inline_html(p, li)
            elif element.name == 'table':
                rows = element.find_all('tr')
                if not rows: continue
                max_cols = max([len(r.find_all(['td', 'th'])) for r in rows])
                table = doc.add_table(rows=len(rows), cols=max_cols)
                table.style = 'Table Grid'
                table.autofit = True
                for i, row in enumerate(rows):
                    cols = row.find_all(['td', 'th'])
                    for j, col in enumerate(cols):
                        if j < max_cols:
                            cell = table.cell(i, j)
                            cell._element.clear_content()
                            p = cell.add_paragraph()
                            DocumentBuilder._process_inline_html(p, col)
                            if row.find('th') or i == 0:
                                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                for run in p.runs: run.bold = True
                                tcPr = cell._element.get_or_add_tcPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:val'), 'clear')
                                shd.set(qn('w:color'), 'auto')
                                shd.set(qn('w:fill'), 'E5E7EB')
                                tcPr.append(shd)

    @staticmethod
    def _process_inline_html(paragraph, element):
        for child in element.children:
            if child.name in ['strong', 'b']: paragraph.add_run(child.get_text()).bold = True
            elif child.name in ['em', 'i']: paragraph.add_run(child.get_text()).italic = True
            elif child.name is None: paragraph.add_run(str(child))
            else: DocumentBuilder._process_inline_html(paragraph, child)

    @staticmethod
    def process_content(doc, raw_text, theme_color=DEFAULT_COLOR, chapter_title=""):
        raw_text = DocumentBuilder._remove_duplicate_header(raw_text, chapter_title)
        lines = raw_text.split('\n')
        clean_lines = []
        in_table = False
        for line in lines:
            line = line.strip()
            if line.startswith('[[CHART:') and line.endswith(']]'):
                data = line.replace('[[CHART:', '').replace(']]', '').strip()
                img = ChartEngine.create_bar_chart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(5.5))
                continue
            if line.startswith('[[GANTT:') and line.endswith(']]'):
                data = line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(6))
                continue
            if line.startswith('[[FLOW:') and line.endswith(']]'):
                data = line.replace('[[FLOW:', '').replace(']]', '').strip()
                img = ChartEngine.create_flowchart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(6.5))
                continue
            if line.startswith('[[MATH:') and line.endswith(']]'):
                data = line.replace('[[MATH:', '').replace(']]', '').strip()
                img = ChartEngine.create_math_image(data, theme_color)
                if img: 
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.add_run().add_picture(img, width=Inches(2.5))
                continue
            if line.startswith('|'):
                if not in_table:
                    if clean_lines and clean_lines[-1] != "": clean_lines.append("")
                    in_table = True
            else: in_table = False
            clean_lines.append(line)
            
        md_text = "\n".join(clean_lines)
        html = markdown.markdown(md_text, extensions=['tables'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)

    @staticmethod
    def create_cover(doc, client, project, logo_stream=None, theme_color=DEFAULT_COLOR):
        StyleEngine.apply_document_styles(doc)
        for _ in range(3): doc.add_paragraph()
        if logo_stream:
            try:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(logo_stream, width=Inches(3))
            except Exception: pass
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
        doc.add_paragraph()
        p_name = doc.add_paragraph(project)
        p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_name.runs[0].font.size = Pt(16)
        p_name.runs[0].italic = True
        for _ in range(4): doc.add_paragraph()
        s = doc.add_paragraph(f"Disusun Oleh:\n{WRITER_FIRM_NAME}")
        s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

class ProposalGenerator:
    def __init__(self, kb_instance):
        self.ollama = Client(host=OLLAMA_HOST)
        self.kb = kb_instance
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.firm_api = FirmAPIClient()

    def _fetch_chapter_context(self, chap, client, project, budget, project_goal, project_type, timeline, notes, regulations, firm_data, research_futures):
        try:
            try: global_data = research_futures['profile'].result(timeout=5)
            except Exception: global_data = ""
            
            try: client_news = research_futures['news'].result(timeout=5)
            except Exception: client_news = "Tidak ada berita spesifik."

            try: regulation_data = research_futures['regulations'].result(timeout=5)
            except Exception: regulation_data = "Standar umum."
            
            try: writer_data = research_futures['contact'].result(timeout=5)
            except Exception: writer_data = "Unavailable"
            
            try: collab_data = research_futures['collab'].result(timeout=5)
            except Exception: collab_data = "Unavailable"
            
            try: firm_exp_data = research_futures['firm_exp'].result(timeout=5)
            except Exception: firm_exp_data = "Unavailable"

            structured_row_data = self.kb.get_exact_context(client, project, budget)
            rag_data = self.kb.query(client, project, chap['keywords'])
            
            discovery_notes = "Tidak ada catatan tambahan."
            if notes: discovery_notes = notes
            
            persona = PERSONAS.get(chap.get('id', 'default'), PERSONAS['default'])
            subs = "\n".join([f"- {s}" for s in chap['subs']])
            
            visual_prompt = "Do not force visuals."
            if "visual_intent" in chap:
                if chap['visual_intent'] == "bar_chart": visual_prompt = "Mandatory Data Visual: [[CHART: Judul | Label | Kat 1,10; Kat 2,20]]"
                elif chap['visual_intent'] == "gantt": visual_prompt = f"Mandatory Timeline Visual: [[GANTT: Jadwal Implementasi | Waktu | Task 1,0,2; Task 2,2,4]]. Align with timeline: {timeline}."
                elif chap['visual_intent'] == "flowchart": visual_prompt = "Process visual: [[FLOW: Step 1 -> Step 2 -> Step 3]]."

            extra = ""
            if chap['id'] in ['chap_1', 'c_chap_1', 'c_chap_2']:
                extra = f"[MANDATORY] Integrate historical evidence of {client} and {WRITER_FIRM_NAME} using: \nCollaboration: {collab_data}"
            elif chap['id'] in ['chap_7', 'c_chap_10']:
                extra = f"[MANDATORY] Detail factual historical evidence of {WRITER_FIRM_NAME}'s experience: {firm_exp_data}"
            elif chap['id'] in ['chap_9', 'c_chap_12']:
                extra = f"[MANDATORY] Create a Markdown Table for Contact Information using: {writer_data}"
            elif chap['id'] == 'c_chap_6':
                extra = f"[MANDATORY] You MUST base your phases EXACTLY on this Firm Methodology: {firm_data['methodology']}."
            elif chap['id'] == 'c_chap_10':
                extra = f"[MANDATORY] You MUST use this exact Team Structure: {firm_data['team']}."
            elif chap['id'] == 'c_chap_11':
                extra = f"[MANDATORY] You MUST state these exact Commercial Rules: {firm_data['commercial']}."

            prompt = PROPOSAL_SYSTEM_PROMPT.format(
                client=client, writer_firm=WRITER_FIRM_NAME, persona=persona,
                global_data=global_data, client_news=client_news, regulation_data=regulation_data,
                structured_row_data=structured_row_data,
                project_goal=project_goal, project_type=project_type, timeline=timeline, discovery_notes=discovery_notes,
                firm_methodology=firm_data['methodology'], firm_team=firm_data['team'], firm_commercial=firm_data['commercial'],
                rag_data=rag_data, visual_prompt=visual_prompt, extra_instructions=extra,
                chapter_title=chap['title'], sub_chapters=subs, length_intent=chap.get('length_intent', 'Be very concise.')
            )

            return {"prompt": prompt, "success": True}
            
        except Exception as e:
            return {"prompt": "", "success": False, "error": str(e)}

    def run(self, client, project, budget=None, service_type="Consulting", project_goal="Improvement", project_type="Implementation", timeline="TBD", notes="", regulations=""):
        logger.info(f"Starting Generation: {client} | Mode Demo: {DEMO_MODE}")
        
        active_structure = CONSULTING_STRUCTURE if service_type == "Consulting" else TRAINING_STRUCTURE
        firm_data = self.firm_api.get_project_standards(project_type)
        
        # Regex yang lebih cerdas: Menghapus Legal Entitas (PT/CV/Tbk) dan Indikator Cabang
        # Tapi TETAP menyisakan nama wilayah jika itu bagian dari nama perusahaan (seperti Garuda Indonesia)
        clean_regex = r'\b(Cabang|Branch|Region|Area|Tbk)\b.*$|^(PT\.|PT\s+|CV\.|CV\s+)'
        
        base_client = re.sub(clean_regex, '', client, flags=re.IGNORECASE).strip()
        base_firm = re.sub(clean_regex, '', WRITER_FIRM_NAME, flags=re.IGNORECASE).strip()

        research_futures = {
            'profile': self.io_pool.submit(Researcher.get_entity_profile, base_client),
            'contact': self.io_pool.submit(Researcher.get_contact_details, WRITER_FIRM_NAME),
            'collab': self.io_pool.submit(Researcher.get_collaboration_data, base_client, base_firm),
            'firm_exp': self.io_pool.submit(Researcher.get_firm_experience, base_firm, project),
            'news': self.io_pool.submit(Researcher.get_latest_client_news, base_client), 
            'regulations': self.io_pool.submit(Researcher.get_regulatory_data, regulations)
        }
        logo_future = self.io_pool.submit(LogoManager.get_logo_and_color, base_client) 
        
        context_futures = {}
        for chap in active_structure:
            context_futures[chap['id']] = self.io_pool.submit(
                self._fetch_chapter_context, chap, client, project, budget, project_goal, project_type, timeline, notes, regulations, firm_data, research_futures
            )

        try: logo_stream, theme_color = logo_future.result(timeout=8)
        except Exception: logo_stream, theme_color = None, DEFAULT_COLOR

        doc = Document()
        DocumentBuilder.create_cover(doc, client, project, logo_stream, theme_color)
        
        for i, chap in enumerate(active_structure):
            ctx = context_futures[chap['id']].result()
            if ctx['success']:
                try:
                    res = self.ollama.chat(
                        model=LLM_MODEL, 
                        messages=[{'role': 'system', 'content': ctx['prompt']}, {'role': 'user', 'content': f"Write content for {chap['title']}."}],
                        options={'num_ctx': 4096}  
                    )
                    h = doc.add_heading(chap['title'], level=1)
                    h.runs[0].font.color.rgb = RGBColor(*theme_color)
                    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    h.paragraph_format.space_before = Pt(0)
                    h.paragraph_format.space_after = Pt(18)
                    h.paragraph_format.keep_with_next = True 
                    
                    DocumentBuilder.process_content(doc, res['message']['content'], theme_color, chap['title'])
                    if i < len(active_structure) - 1: doc.add_page_break()
                except Exception as e: logger.error(f"Error {chap['title']}: {e}")

        return doc, f"Proposal_{client}_{project}".replace(" ", "_")