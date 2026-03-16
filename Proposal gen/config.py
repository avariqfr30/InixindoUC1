"""
Configuration and constants for the Proposal Generator Engine.
"""
import os

# =====================================================================
# SYSTEM MODE & API ADAPTER
# =====================================================================
DEMO_MODE = True 
FIRM_API_URL = "https://api.perusahaan-anda.com/v1" 
API_AUTH_TOKEN = "isi_token_disini_nanti" # <-- Paste real token here

# --- CREDENTIALS & HOSTS ---
SERPER_API_KEY = "YOUR_SERPER_API_KEY"    # <-- Paste real Serper key here
OLLAMA_HOST = "http://127.0.0.1:11434"

# --- MODELS & DB ---
LLM_MODEL = "gpt-oss:120b-cloud" 
EMBED_MODEL = "bge-m3:latest"
DB_URI = "sqlite:///projects.db" 

# --- FIRM IDENTITY ---
WRITER_FIRM_NAME = "Inixindo Jogja" 
DEFAULT_COLOR = (0, 51, 102)

DATA_MAPPING = {
    "entity": "Client Entity",
    "topic": "Strategic Initiative",
    "budget": "Investment Estimation"
}

# =====================================================================
# KNOWLEDGE PATTERN LIBRARY
# =====================================================================
SMART_SUGGESTIONS = {
    "banking": {
        "keywords": ["bank", "bca", "bri", "mandiri", "finance", "fintech", "payment", "kredit"],
        "regulations": ["POJK No. 4/POJK.05/2021", "ISO 27001", "PCI-DSS", "UU PDP"]
    },
    "government": {
        "keywords": ["kementerian", "dinas", "pemprov", "pemkab", "bumn", "publik", "layanan"],
        "regulations": ["Perpres SPBE No 95 Tahun 2018", "SNI ISO 37001", "UU PDP"]
    },
    "healthcare": {
        "keywords": ["rumah sakit", "klinik", "bpjs", "kesehatan", "medis"],
        "regulations": ["Permenkes No 24 Tahun 2022", "HIPAA", "ISO 27001"]
    },
    "general_it": {
        "keywords": ["infrastruktur", "cloud", "aplikasi", "software", "jaringan", "data center"],
        "regulations": ["ITIL v4", "COBIT 2019", "TOGAF", "ISO 20000", "ISO 27001"]
    }
}

# =====================================================================
# MOCK API DATA
# =====================================================================
MOCK_FIRM_PROFILE = {
    "contact_info": "Kantor Pusat Inixindo Jogja\nJl. Kenari No. 69, Yogyakarta\nEmail: info@inixindo.id\nTelp: (0274) 515448",
    "portfolio_highlights": "Pengalaman di Transformasi TI, IT Master Plan, DevSecOps, dan ISO/POJK."
}

MOCK_FIRM_STANDARDS = {
    "Diagnostic": {
        "methodology": "1. Data Gathering\n2. As-Is Analysis\n3. Gap Analysis\n4. Diagnostic Report.",
        "team": "1x Principal Consultant, 2x Senior BA.",
        "commercial": "Fixed-fee. 50% Upfront, 50% on Report delivery."
    },
    "Strategic": {
        "methodology": "1. Visioning\n2. Target Operating Model\n3. Roadmap\n4. Board Presentation.",
        "team": "1x Project Director, 1x Enterprise Architect.",
        "commercial": "Retainer or Fixed-fee based on milestones."
    },
    "Transformation": {
        "methodology": "1. Readiness Assessment\n2. Phased Rollout\n3. Change Management\n4. Hypercare.",
        "team": "1x Program Manager, 2x Change Experts, 3x Tech Leads.",
        "commercial": "Time & Materials (T&M) over 12-18 months."
    },
    "Implementation": {
        "methodology": "1. Design\n2. Configuration\n3. UAT\n4. Go-Live\n5. Handover.",
        "team": "1x PM, 1x Lead Engineer, 3x Implementers.",
        "commercial": "Milestones: 20% Kickoff, 40% UAT, 30% Go-Live, 10% Handover."
    }
}

# =====================================================================
# PROPOSAL STRUCTURE (CONCISE & DENSE TARGET)
# =====================================================================
UNIVERSAL_STRUCTURE = [
    {
        "id": "c_1", "title": "BAB I – KONTEKS ORGANISASI",
        "subs": ["1.1 Latar Belakang Singkat", "1.2 Visi & Objektif"],
        "keywords": "context background vision strategy",
        "length_intent": "Target: 150 words. Write directly and concisely using bullet points. Focus purely on facts from OSINT."
    },
    {
        "id": "c_2", "title": "BAB II – PERMASALAHAN",
        "subs": ["2.1 Tantangan Utama", "2.2 Dampak Bisnis"],
        "keywords": "problem pain points bottleneck issue",
        "length_intent": "Target: 200 words. Provide a direct, unpadded Root Cause Analysis. List impacts using bullet points."
    },
    {
        "id": "c_3", "title": "BAB III – SOLUSI & PENDEKATAN",
        "subs": ["3.1 Arsitektur Solusi", "3.2 Nilai Tambah"],
        "keywords": "solution approach architecture",
        "length_intent": "Target: 200 words. Explain technical implementation practically without academic theory."
    },
    {
        "id": "c_4", "title": "BAB IV – KERANGKA KERJA & KEPATUHAN",
        "subs": ["4.1 Framework Terpilih", "4.2 Pemenuhan Regulasi"],
        "keywords": "framework iso cobit regulation compliance",
        "length_intent": "Target: 150 words. State exactly how the framework maps to the client's needs."
    },
    {
        "id": "c_5", "title": "BAB V – METODOLOGI & WAKTU",
        "subs": ["5.1 Fase Pelaksanaan", "5.2 Jadwal Estimasi"],
        "keywords": "methodology timeline schedule phase",
        "length_intent": "Target: 200 words. Highly structured, step-by-step methodology breakdown. No filler.",
        "visual_intent": "gantt"
    },
    {
        "id": "c_6", "title": "BAB VI – TIM & KAPABILITAS",
        "subs": ["6.1 Struktur Tim", "6.2 Portofolio Firm"],
        "keywords": "team expert role portfolio",
        "length_intent": "Target: 150 words. Confident list of roles and past successes."
    },
    {
        "id": "c_7", "title": "BAB VII – ESTIMASI BIAYA & PENUTUP",
        "subs": ["7.1 Investasi", "7.2 Kesimpulan & Langkah Lanjut"],
        "keywords": "budget cost investment closing",
        "length_intent": "Target: 200 words. Include a markdown table for pricing. Provide a decisive 2-sentence closing."
    }
]

PERSONAS = {
    "default": "Principal Management Consultant",
    "c_1": "Senior Business Analyst",
    "c_3": "Chief Technology Officer",
    "c_7": "Commercial Director"
}

PROPOSAL_SYSTEM_PROMPT = """
You are an elite Principal Consultant at {writer_firm}.
Your audience is the Executive Board of {client}. Adopt this persona: {persona}.

--- STRICT WRITING RULES ---
1. NO FLUFF: Write strictly "singkat, padat, tidak bertele-tele" (concise, dense, non-theoretical).
2. FORMATTING: Maximize the use of bullet points and short paragraphs. Avoid dense blocks of text.
3. TONE: Professional, objective, direct, and highly persuasive in Indonesian. Do not use academic filler.
4. SYNTHESIS: Integrate OSINT data naturally as facts, do not mention "berdasarkan sumber online".

--- CONTEXT DATA ---
Global OSINT Data: {global_data}
Client News: {client_news}
Regulatory Data: {regulation_data}
Historical Data: {structured_row_data}
Semantic RAG: {rag_data}

--- DYNAMIC INSTRUCTIONS ---
{extra_instructions}

--- TASK ---
Write "{chapter_title}" ensuring it meets the length intent: {length_intent}
Include these H2 (##) exactly:
{sub_chapters}

{visual_prompt}
DO NOT write greetings or introductions. Output strictly the chapter content.
"""