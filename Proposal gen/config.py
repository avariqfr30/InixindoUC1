# config.py
import os

# --- CREDENTIALS & HOSTS ---
GOOGLE_API_KEY = "API_KEY"
GOOGLE_CX_ID = "CX_ID"
OLLAMA_HOST = "http://127.0.0.1:11434"

# --- MODELS & DB ---
LLM_MODEL = "gpt-oss:120b-cloud" 
EMBED_MODEL = "bge-m3:latest"
DB_FILE = "db.csv"

# --- FIRM IDENTITY ---
WRITER_FIRM_NAME = "Inixindo Jogja" 
DEFAULT_COLOR = (0, 51, 102)

# --- PROMPTS ---
PROPOSAL_SYSTEM_PROMPT = """
You are a Senior Consultant for {client} working at {writer_firm}. 
ROLE: {persona}.

=== LIVE WEB CONTEXT (PROFILING) ===
{web_data}

=== EXACT PROJECT REQUIREMENTS (STRUCTURED DATA) ===
{structured_row_data}

=== HISTORICAL VECTOR CONTEXT ===
{rag_data}

MANDATORY RULES:
1. DO NOT repeat the Chapter Title in your output.
2. WRITE EXTENSIVELY (Word-Heavy).
3. Ground your entire response ONLY in the Exact Project Requirements provided above. Do not invent methodologies or features that are not listed in the structural data.
4. {visual_prompt}
5. {extra_instructions}

WRITE CONTENT FOR '{chapter_title}' covering:
{sub_chapters}
"""

# --- SCHEMAS (Mapped to Official Draft Proposal.pdf) ---
PROPOSAL_STRUCTURE = [
    {
        "id": "chap_1", "title": "BAB I – LATAR BELAKANG",
        "subs": [
            "1.1 Gambaran Umum Tantangan Organisasi", 
            "1.2 Perubahan Lingkungan Bisnis & Teknologi", 
            "1.3 Kesenjangan Kompetensi / Kematangan Digital", 
            "1.4 Dampak Jika Masalah Tidak Ditangani", 
            "1.5 Urgensi Program Pelatihan &/atau Konsultasi"
        ],
        "keywords": "problem pain points business environment gap urgency",
        "visual_intent": "bar_chart",
        "length_intent": "Comprehensive and highly detailed. Thoroughly explain the background and urgency (approx. 500 words)."
    },
    {
        "id": "chap_2", "title": "BAB II – MAKSUD DAN TUJUAN PROGRAM",
        "subs": [
            "2.1 Maksud Program", 
            "2.2 Tujuan Umum", 
            "2.3 Tujuan Khusus", 
            "2.4 Sasaran Program (Individu / Tim / Organisasi)", 
            "2.5 Ruang Lingkup Program"
        ],
        "keywords": "objectives scope targets",
        "length_intent": "Clear, structured, and descriptive. Elaborate on each objective using bullet points where necessary (approx. 300 words)."
    },
    {
        "id": "chap_3", "title": "BAB III – SOLUSI YANG DITAWARKAN",
        "subs": [
            "3.1 Pendekatan Solusi (Training, Consulting, atau Terintegrasi)", 
            "3.2 Kerangka Solusi Berbasis Kebutuhan Klien", 
            "3.3 Posisi Solusi dalam Peta Transformasi Klien", 
            "3.4 Nilai Tambah & Diferensiasi Solusi", 
            "3.5 Solusi Pelatihan: Konsep, Learning Path, Metodologi, Metode Pelaksanaan, Output", 
            "3.6 Solusi Konsultan: Pendekatan, Metodologi Kerja, Deliverables, Keterlibatan Tim"
        ],
        "keywords": "solution framework training consulting methodology",
        "visual_intent": "flowchart",
        "length_intent": "Highly detailed, technical, and methodical. Provide deep explanations of the framework and methodologies. Use clear sub-headings (approx. 800 words)."
    },
    {
        "id": "chap_4", "title": "BAB IV – RENCANA AKSI & IMPLEMENTASI",
        "subs": [
            "4.1 Tahapan Pelaksanaan Program", 
            "4.2 Timeline / Jadwal Kegiatan", 
            "4.3 Peran dan Tanggung Jawab (RACI)", 
            "4.4 Mekanisme Koordinasi & Pelaporan", 
            "4.5 Manajemen Risiko Pelaksanaan"
        ],
        "keywords": "action plan timeline raci risk management",
        "visual_intent": "gantt",
        "length_intent": "Structured timeline and responsibilities. Use strict Markdown tables for the RACI matrix and Timeline (approx. 400 words)."
    },
    {
        "id": "chap_5", "title": "BAB V – OUTPUT, OUTCOME, DAN INDIKATOR KEBERHASILAN",
        "subs": [
            "5.1 Output Program", 
            "5.2 Outcome Jangka Pendek", 
            "5.3 Outcome Jangka Menengah & Panjang", 
            "5.4 KPI Keberhasilan Program", 
            "5.5 Mekanisme Evaluasi & Perbaikan"
        ],
        "keywords": "kpi output outcome evaluation",
        "length_intent": "Impact-focused and highly descriptive. Explain the metrics and outcomes clearly (approx. 350 words)."
    },
    {
        "id": "chap_6", "title": "BAB VI – TIM PELAKSANA",
        "subs": [
            "6.1 Struktur Tim", 
            "6.2 Peran & Kompetensi Tim", 
            "6.3 Pengalaman Relevan Tim"
        ],
        "keywords": "team structure roles competence",
        "length_intent": "Professional profiles and team hierarchy. Use bullet points for competencies (approx. 200 words)."
    },
    {
        "id": "chap_7", "title": "BAB VII – PORTOFOLIO & PENGALAMAN",
        "subs": [
            "7.1 Profil Perusahaan", 
            "7.2 Portofolio Proyek / Pelatihan", 
            "7.3 Klien & Mitra Strategis", 
            "7.4 Testimoni"
        ],
        "keywords": "portfolio profile credibility Inixindo",
        "length_intent": "Persuasive and credential-focused. Write confidently about the firm's experience (approx. 250 words)."
    },
    {
        "id": "chap_8", "title": "BAB VIII – SKEMA BIAYA & INVESTASI",
        "subs": [
            "8.1 Ruang Lingkup Biaya", 
            "8.2 Skema Pembiayaan", 
            "8.3 Value vs Investment", 
            "8.4 Ketentuan Pembayaran"
        ],
        "keywords": "investment cost pricing ROI",
        "length_intent": "Highly concise. Use strict Markdown tables for cost breakdowns and bullet points for terms. Do not write filler text (approx. 200 words)."
    },
    {
        "id": "chap_9", "title": "BAB IX – PENUTUP",
        "subs": [
            "9.1 Komitmen dan Pernyataan Kesanggupan", 
            "9.2 Harapan Kerja Sama", 
            "9.3 Kontak & Tindak Lanjut"
        ],
        "keywords": "closing contact commitment",
        "length_intent": "Brief and impactful. Direct to the point (under 100 words)."
    }
]

PERSONAS = {
    "chap_1": "CEO & Board (Strategic, Urgent, Big Picture)",
    "chap_3": "CTO/CIO (Technical, Methodical, Safe)",
    "chap_8": "CFO/Procurement (Financial, ROI-focused, Strict)",
    "default": "Project Manager (Operational, Clear, Structured)"
}