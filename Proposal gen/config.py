# config.py
import os

# --- CREDENTIALS & HOSTS ---
GOOGLE_API_KEY = "API_KEY"
GOOGLE_CX_ID = "CX_KEY"
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
CONTEXT: {global_data} \n {rag_data}

MANDATORY RULES:
1. DO NOT repeat the Chapter Title in your output.
2. WRITE EXTENSIVELY (Word-Heavy).
3. {visual_prompt}
4. {extra_instructions}

WRITE CONTENT FOR '{chapter_title}' covering:
{sub_chapters}
"""

# --- SCHEMAS ---
PROPOSAL_STRUCTURE = [
    {
        "id": "chap_1", "title": "BAB I - LATAR BELAKANG",
        "subs": ["1.1 Tantangan Organisasi", "1.2 Tren & Konteks Bisnis", "1.3 Gap Kompetensi", "1.4 Dampak Resiko", "1.5 Urgensi Solusi"],
        "keywords": "pain points challenges market trends stats",
        "visual_intent": "bar_chart" 
    },
    {
        "id": "chap_2", "title": "BAB II - MAKSUD DAN TUJUAN",
        "subs": ["2.1 Maksud Program", "2.2 Tujuan Umum", "2.3 Tujuan Khusus", "2.4 Sasaran Peserta", "2.5 Ruang Lingkup"],
        "keywords": "objectives kpi goals targets"
    },
    {
        "id": "chap_3", "title": "BAB III - SOLUSI & PENDEKATAN",
        "subs": ["3.1 Metodologi Solusi", "3.2 Framework Pelaksanaan", "3.3 Nilai Tambah", "3.4 Roadmap Transformasi", "3.5 Detail Teknis"],
        "keywords": "solution methodology framework technical approach",
        "visual_intent": "flowchart" 
    },
    {
        "id": "chap_4", "title": "BAB IV - RENCANA IMPLEMENTASI",
        "subs": ["4.1 Tahapan Kerja", "4.2 Timeline Proyek", "4.3 Matrix Tanggung Jawab (RACI)", "4.4 Manajemen Resiko"],
        "keywords": "implementation timeline project management raci",
        "visual_intent": "gantt"
    },
    {
        "id": "chap_5", "title": "BAB V - OUTPUT & OUTCOME",
        "subs": ["5.1 Deliverables", "5.2 Dampak Jangka Pendek", "5.3 Dampak Jangka Panjang", "5.4 KPI Keberhasilan"],
        "keywords": "outcomes deliverables impact metrics"
    },
    {
        "id": "chap_6", "title": "BAB VI - TIM PELAKSANA",
        "subs": ["6.1 Struktur Organisasi Tim", "6.2 Profil Tenaga Ahli", "6.3 Kualifikasi Tim"],
        "keywords": "team structure expert profiles",
    },
    {
        "id": "chap_7", "title": "BAB VII - PORTOFOLIO PERUSAHAAN",
        "subs": ["7.1 Tentang Inixindo Jogja", "7.2 Pengalaman Relevan", "7.3 Daftar Klien Strategis"],
        "keywords": "company profile case studies portfolio Inixindo Jogja"
    },
    {
        "id": "chap_8", "title": "BAB VIII - INVESTASI",
        "subs": ["8.1 Komponen Biaya", "8.2 Rincian Investasi", "8.3 Term of Payment"],
        "keywords": "investment cost pricing budget rupiah"
    },
    {
        "id": "chap_9", "title": "BAB IX - PENUTUP",
        "subs": ["9.1 Komitmen", "9.2 Next Steps", "9.3 Kontak & Tindak Lanjut"],
        "keywords": "closing contact commitment"
    }
]

PERSONAS = {
    "chap_1": "CEO & Board (Strategic, Urgent, Big Picture)",
    "chap_3": "CTO/CIO (Technical, Methodical, Safe)",
    "chap_8": "CFO/Procurement (Financial, ROI-focused, Strict)",
    "default": "Project Manager (Operational, Clear, Structured)"
}