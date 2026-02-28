import os
import io
import re
import logging
import requests
import pandas as pd
import chromadb
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from PIL import Image, ImageStat
from difflib import SequenceMatcher

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
from chromadb.config import Settings # Add this line
from chromadb.utils import embedding_functions

# Import configurations
from config import (
    GOOGLE_API_KEY, GOOGLE_CX_ID, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL,
    WRITER_FIRM_NAME, DEFAULT_COLOR, PROPOSAL_STRUCTURE, PERSONAS, 
    PROPOSAL_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Manages the local vector database using ChromaDB and Ollama embeddings.
    Handles data ingestion from CSV and context retrieval via RAG.
    """
    def __init__(self, db_file):
        self.db_file = db_file
        
        # Initialize client with telemetry disabled
        self.chroma = chromadb.Client(Settings(anonymized_telemetry=False)) 
        
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings", 
            model_name=EMBED_MODEL
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", 
            embedding_function=self.embed_fn
        )
        self.df = None
        self.refresh_data()

    def refresh_data(self):
        if not os.path.exists(self.db_file): 
            logger.warning(f"Database file {self.db_file} not found.")
            return False
            
        try:
            self.df = pd.read_csv(self.db_file)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            existing = self.collection.get()['ids']
            if existing: 
                self.collection.delete(existing)
            
            ids, docs, metas = [], [], []
            for idx, row in self.df.iterrows():
                client = row.get('Client Entity')
                project = row.get('Strategic Initiative')
                context = row.get('Strategic Context & Pain Points')
                text_rep = f"Client: {client} | Project: {project} | Context: {context}"
                
                ids.append(str(idx))
                docs.append(text_rep)
                metas.append(row.to_dict())
                
            if ids: 
                self.collection.add(documents=docs, metadatas=metas, ids=ids)
            return True
            
        except Exception as e: 
            logger.error(f"Error refreshing KnowledgeBase data: {e}")
            return False

    def query(self, client, project, context_keywords=None):
        try:
            q_text = f"{project} implementation for {client} {context_keywords or ''}"
            res = self.collection.query(query_texts=[q_text], n_results=2)
            if res['documents']: 
                return "\n".join(res['documents'][0])
        except Exception as e: 
            logger.error(f"Error querying vector DB: {e}")
            
        return ""

class Researcher:
    @staticmethod
    def search(query, limit=2):
        if "YOUR_GOOGLE" in GOOGLE_API_KEY: 
            return None
            
        try:
            params = {
                'q': query, 
                'key': GOOGLE_API_KEY, 
                'cx': GOOGLE_CX_ID, 
                'num': limit, 
                'gl': 'id'
            }
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e: 
            logger.error(f"Google Search API failed for query '{query}': {e}")
            return None

    @staticmethod
    def fetch_page_content(url):
        """Fetches and extracts visible text from a given URL."""
        try:
            # Add headers to prevent basic bot-blocking
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Extract text from paragraphs and headers
            texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            content = ' '.join([t.get_text(strip=True) for t in texts])
            
            # Limit characters to avoid blowing up the LLM context window
            return content[:4000] 
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return ""

    @staticmethod
    def get_entity_profile(entity_name):
        res = Researcher.search(f"{entity_name} official company profile about us", limit=2)
        if not res or 'items' not in res: 
            return f"{entity_name} (Contact info lookup failed)"
        
        # Fetch actual text from the top URL instead of just the snippet
        top_url = res['items'][0]['link']
        page_content = Researcher.fetch_page_content(top_url)
        
        if page_content:
            return f"Profile data from {top_url}:\n{page_content}"
        
        # Fallback to snippets if scraping fails
        return "\n".join([i.get('snippet', '') for i in res['items']])

    @staticmethod
    def get_contact_details(entity_name):
        query = f"{entity_name} alamat nomor telepon email contact details"
        res = Researcher.search(query, limit=3)
        if not res or 'items' not in res: 
            return f"{entity_name} (Details unavailable in search)"
        return "\n".join([i.get('snippet', '') for i in res['items']])

class LogoManager:
    """
    Fetches client logos and extracts the dominant theme color to style the document.
    """
    @staticmethod
    def get_logo_and_color(client_name):
        if "YOUR_GOOGLE" in GOOGLE_API_KEY: 
            return None, DEFAULT_COLOR
            
        try:
            params = {
                'q': f"{client_name} logo png transparent", 
                'key': GOOGLE_API_KEY, 
                'cx': GOOGLE_CX_ID, 
                'num': 1, 
                'searchType': 'image'
            }
            res = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=4).json()
            
            if 'items' in res:
                img_resp = requests.get(res['items'][0]['link'], timeout=5)
                if img_resp.status_code == 200:
                    stream = io.BytesIO(img_resp.content)
                    try:
                        img = Image.open(stream).convert('RGB')
                        img.thumbnail((150, 150))
                        dom_color = tuple(map(int, ImageStat.Stat(img).mean[:3]))
                        stream.seek(0)
                        return stream, dom_color
                    except Exception as e:
                        logger.error(f"Failed to process image colors: {e}")
                        stream.seek(0)
                        return stream, DEFAULT_COLOR
        except Exception as e: 
            logger.error(f"Failed to fetch logo for {client_name}: {e}")
            
        return None, DEFAULT_COLOR

class StyleEngine:
    """
    Applies base document margins, typography, and paragraph spacing.
    """
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
    """
    Handles the generation of matplotlib visuals for the proposal.
    All charts return in-memory image streams (BytesIO).
    """
    @staticmethod
    def _get_plt_color(theme_color):
        return tuple(c/255 for c in theme_color)

    @staticmethod
    def create_bar_chart(data_str, theme_color):
        try:
            labels, values = [], []
            for p in data_str.split(';'):
                if ',' in p:
                    l, v = p.split(',')
                    labels.append(l.strip())
                    values.append(float(re.sub(r'[^\d.]', '', v)))
                    
            if not labels: 
                return None

            plt.figure(figsize=(6.5, 3.5))
            plt.bar(labels, values, color=ChartEngine._get_plt_color(theme_color), 
                   alpha=0.9, width=0.6, zorder=3, edgecolor='white', linewidth=0.5)
            plt.title("Analisis Data Penunjang", fontsize=11, fontweight='bold', pad=15, color='#333333')
            plt.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tick_params(axis='x', rotation=15, labelsize=9)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception as e: 
            logger.error(f"Bar chart generation failed: {e}")
            return None

    @staticmethod
    def create_gantt_chart(data_str, theme_color):
        try:
            tasks = []
            for p in data_str.split(';'):
                parts = p.split(',')
                if len(parts) >= 3:
                    tasks.append({
                        "task": parts[0].strip(),
                        "start": float(re.sub(r'[^\d.]', '', parts[1])),
                        "dur": float(re.sub(r'[^\d.]', '', parts[2]))
                    })
                    
            if not tasks: 
                return None
            
            tasks = tasks[::-1]
            names = [t['task'] for t in tasks]
            starts = [t['start'] for t in tasks]
            durs = [t['dur'] for t in tasks]
            
            fig, ax = plt.subplots(figsize=(7.5, max(3, len(tasks)*0.7)))
            
            for i, (name, start, dur) in enumerate(zip(names, starts, durs)):
                rect = patches.FancyBboxPatch((start, i-0.25), dur, 0.5, 
                                            boxstyle="round,pad=0.02,rounding_size=0.1",
                                            ec="none", fc=ChartEngine._get_plt_color(theme_color), 
                                            alpha=0.85)
                ax.add_patch(rect)

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel("Timeline (Bulan/Minggu)", fontsize=9, fontweight='bold')
            ax.set_title("Rencana Implementasi", fontsize=12, fontweight='bold', pad=15)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            ax.set_xlim(0, max([t['start'] + t['dur'] for t in tasks]) + 1)
            ax.set_ylim(-0.5, len(names)-0.5)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception as e: 
            logger.error(f"Gantt chart generation failed: {e}")
            return None

    @staticmethod
    def create_flowchart(data_str, theme_color):
        try:
            steps = [s.strip() for s in data_str.split('->')]
            if len(steps) < 2: 
                return None
            
            steps = ["\n".join(textwrap.wrap(s, width=18)) for s in steps]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis('off')
            
            n_steps = len(steps)
            x_pos = [i * 2.5 for i in range(n_steps)]
            y_pos = 0.5
            
            for i in range(n_steps - 1):
                ax.annotate("", xy=(x_pos[i+1]-1.0, y_pos), xytext=(x_pos[i]+1.0, y_pos),
                           arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5, mutation_scale=15))

            for i, step in enumerate(steps):
                box = patches.FancyBboxPatch((x_pos[i]-1.0, y_pos-0.4), 2.0, 0.8,
                                           boxstyle="round,pad=0.1,rounding_size=0.2",
                                           fc=ChartEngine._get_plt_color(theme_color),
                                           ec="#2c3e50", alpha=0.9, zorder=2)
                ax.add_patch(box)
                ax.text(x_pos[i], y_pos, step, ha="center", va="center", 
                       size=9, color="white", fontweight='bold', zorder=3)

            ax.set_xlim(-1.2, (n_steps-1)*2.5 + 1.2)
            ax.set_ylim(0, 1)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=200, transparent=True)
            plt.close()
            img.seek(0)
            return img
        except Exception as e: 
            logger.error(f"Flowchart generation failed: {e}")
            return None

    @staticmethod
    def create_math_image(latex_str, theme_color):
        try:
            fig = plt.figure(figsize=(6, 0.8))
            clean_tex = latex_str.replace('*', r'\times').replace('=', r'\=').strip()
            plt.text(0.5, 0.5, f"${clean_tex}$", fontsize=16, ha='center', va='center', 
                    color='#222222', fontweight='normal')
            plt.axis('off')
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=200, transparent=True)
            plt.close()
            img.seek(0)
            return img
        except Exception as e: 
            logger.error(f"Math image generation failed: {e}")
            return None

class DocumentBuilder:
    """
    Parses Markdown/HTML from the LLM and translates it into native docx elements.
    """
    @staticmethod
    def _remove_duplicate_header(raw_text, title):
        lines = raw_text.strip().split('\n')
        if not lines: 
            return raw_text
            
        first_line = lines[0].strip().replace('#', '').strip()
        ratio = SequenceMatcher(None, first_line.lower(), title.lower()).ratio()
        
        if ratio > 0.6 or first_line.lower() in title.lower():
            return "\n".join(lines[1:]).strip()
        return raw_text

    @staticmethod
    def parse_html_to_docx(doc, html_content, theme_color):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup.children:
            if element.name is None: 
                continue
            
            if element.name in ['h1', 'h2', 'h3']:
                level = int(element.name[1])
                p = doc.add_heading(element.get_text().strip(), level=level)
                for run in p.runs:
                    run.font.color.rgb = RGBColor(*theme_color)
                    run.font.name = 'Arial'
                    run.font.size = Pt(14 if level == 2 else 12)
            
            elif element.name == 'p':
                text = element.get_text().strip()
                if not text: 
                    continue
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
                if not rows: 
                    continue
                    
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
                                for run in p.runs: 
                                    run.bold = True
                                    
                                tcPr = cell._element.get_or_add_tcPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:val'), 'clear')
                                shd.set(qn('w:color'), 'auto')
                                shd.set(qn('w:fill'), 'E5E7EB')
                                tcPr.append(shd)

    @staticmethod
    def _process_inline_html(paragraph, element):
        for child in element.children:
            if child.name in ['strong', 'b']:
                paragraph.add_run(child.get_text()).bold = True
            elif child.name in ['em', 'i']:
                paragraph.add_run(child.get_text()).italic = True
            elif child.name is None:
                paragraph.add_run(str(child))
            else:
                DocumentBuilder._process_inline_html(paragraph, child)

    @staticmethod
    def process_content(doc, raw_text, theme_color=DEFAULT_COLOR, chapter_title=""):
        raw_text = DocumentBuilder._remove_duplicate_header(raw_text, chapter_title)
        lines = raw_text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Visual Processing Tokens
            if line.startswith('[[CHART:') and line.endswith(']]'):
                data = line.replace('[[CHART:', '').replace(']]', '').strip()
                img = ChartEngine.create_bar_chart(data, theme_color)
                if img: 
                    doc.add_paragraph().add_run().add_picture(img, width=Inches(5.5))
                continue
            
            if line.startswith('[[GANTT:') and line.endswith(']]'):
                data = line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img: 
                    doc.add_paragraph().add_run().add_picture(img, width=Inches(6))
                continue
                
            if line.startswith('[[FLOW:') and line.endswith(']]'):
                data = line.replace('[[FLOW:', '').replace(']]', '').strip()
                img = ChartEngine.create_flowchart(data, theme_color)
                if img: 
                    doc.add_paragraph().add_run().add_picture(img, width=Inches(6.5))
                continue

            if line.startswith('[[MATH:') and line.endswith(']]'):
                data = line.replace('[[MATH:', '').replace(']]', '').strip()
                img = ChartEngine.create_math_image(data, theme_color)
                if img: 
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.add_run().add_picture(img, width=Inches(2.5))
                continue
                
            clean_lines.append(line)
        
        md_text = "\n".join(clean_lines)
        html = markdown.markdown(md_text, extensions=['tables'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)

    @staticmethod
    def create_cover(doc, client, project, logo_stream=None, theme_color=DEFAULT_COLOR):
        StyleEngine.apply_document_styles(doc)
        
        for _ in range(3): 
            doc.add_paragraph()
            
        if logo_stream:
            try:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(logo_stream, width=Inches(3))
            except Exception as e: 
                logger.error(f"Failed to add cover logo: {e}")
        
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

        for _ in range(4): 
            doc.add_paragraph()
            
        s = doc.add_paragraph(f"Disusun Oleh:\n{WRITER_FIRM_NAME}")
        s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

class ProposalGenerator:
    """
    Orchestrates the data gathering, LLM prompting, and document assembly.
    """
    def __init__(self, kb_instance):
        self.ollama = Client(host=OLLAMA_HOST)
        self.kb = kb_instance
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def _fetch_chapter_context(self, chap, client, project, global_future, writer_future):
        try:
            try: 
                global_data = global_future.result(timeout=10)
            except Exception: 
                global_data = ""
            
            try: 
                writer_data = writer_future.result(timeout=10)
            except Exception: 
                writer_data = "Unavailable"

            rag_data = self.kb.query(client, project, chap['keywords'])
            persona = PERSONAS.get(chap['id'], PERSONAS['default'])
            subs = "\n".join([f"- {s}" for s in chap['subs']])
            
            visual_prompt = "Do not force visuals."
            if "visual_intent" in chap:
                if chap['visual_intent'] == "bar_chart":
                    visual_prompt = "Supported by: [[CHART: Label,10; Label2,20]]."
                elif chap['visual_intent'] == "gantt":
                    visual_prompt = "Timeline visual: [[GANTT: Phase1,1,2; Phase2,3,4]]."
                elif chap['visual_intent'] == "flowchart":
                    visual_prompt = "Process visual: [[FLOW: Step1 -> Step2]]."

            extra = ""
            if chap['id'] == 'chap_9':
                extra = f"""
                [MANDATORY]
                Create a Markdown Table for Contact Information.
                Use exactly this data found via search for {WRITER_FIRM_NAME}:
                {writer_data}
                Columns: "Office Location", "Phone", "Email".
                Always refer to the writer firm as "{WRITER_FIRM_NAME}", not just "Inixindo".
                """

            prompt = PROPOSAL_SYSTEM_PROMPT.format(
                client=client,
                writer_firm=WRITER_FIRM_NAME,
                persona=persona,
                global_data=global_data,
                rag_data=rag_data,
                visual_prompt=visual_prompt,
                extra_instructions=extra,
                chapter_title=chap['title'],
                sub_chapters=subs
            )

            return {"prompt": prompt, "success": True}
            
        except Exception as e:
            logger.error(f"Failed to fetch context for {chap['id']}: {e}")
            return {"prompt": "", "success": False, "error": str(e)}

    def run(self, client, project):
        logger.info(f"Starting Proposal Generation: Client={client}, Project={project}")
        
        global_future = self.io_pool.submit(Researcher.get_entity_profile, client)
        writer_future = self.io_pool.submit(Researcher.get_contact_details, WRITER_FIRM_NAME)
        logo_future = self.io_pool.submit(LogoManager.get_logo_and_color, client)
        
        context_futures = {}
        for chap in PROPOSAL_STRUCTURE:
            context_futures[chap['id']] = self.io_pool.submit(
                self._fetch_chapter_context, chap, client, project, global_future, writer_future
            )

        try: 
            logo_stream, theme_color = logo_future.result(timeout=8)
        except Exception as e:
            logger.warning(f"Logo retrieval timed out or failed: {e}")
            logo_stream, theme_color = None, DEFAULT_COLOR

        doc = Document()
        DocumentBuilder.create_cover(doc, client, project, logo_stream, theme_color)
        
        for chap in PROPOSAL_STRUCTURE:
            logger.info(f"Generating content for: {chap['title']}")
            ctx = context_futures[chap['id']].result()
            
            if ctx['success']:
                try:
                    res = self.ollama.chat(
                        model=LLM_MODEL, 
                        messages=[
                            {'role': 'system', 'content': ctx['prompt']}, 
                            {'role': 'user', 'content': f"Write content for {chap['title']}."}
                        ],
                        options={'num_ctx': 4096}
                    )
                    
                    h = doc.add_heading(chap['title'], level=1)
                    h.runs[0].font.color.rgb = RGBColor(*theme_color)
                    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    DocumentBuilder.process_content(doc, res['message']['content'], theme_color, chap['title'])
                    doc.add_page_break()
                    
                except Exception as e:
                    logger.error(f"LLM Generation Error for {chap['title']}: {e}")

        filename = f"Proposal_{client}_{project}".replace(" ", "_")
        return doc, filename