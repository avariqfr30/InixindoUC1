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
# PROPOSAL STRUCTURE (TARGET: 20-25 PAGES TOTAL)
# =====================================================================
UNIVERSAL_STRUCTURE = [
    {
        "id": "c_1", "title": "BAB I – KONTEKS KLIEN",
        "subs": ["1.1 Latar Belakang Organisasi Klien", "1.2 Alasan Permintaan Jasa Konsultasi"],
        "keywords": "client context background organizational profile",
        "length_intent": "Jelaskan latar belakang organisasi klien yang menjadi dasar kebutuhan konsultasi, termasuk kondisi internal-eksternal yang relevan. (Target: 700 words)."
    },
    {
        "id": "c_2", "title": "BAB II – PERMASALAHAN",
        "subs": ["2.1 Kebutuhan atau Keinginan Klien", "2.2 Rumusan Masalah yang Harus Diselesaikan"],
        "keywords": "problem statement pain points client needs",
        "length_intent": "Tangkap dengan jelas apa yang benar-benar dibutuhkan atau diinginkan klien serta konteks masalah inti yang harus diselesaikan. (Target: 750 words)."
    },
    {
        "id": "c_3", "title": "BAB III – KLASIFIKASI KEBUTUHAN",
        "subs": ["3.1 Klasifikasi Problem/Opportunity/Directive", "3.2 Tujuan Utama dan Jenis Proyek"],
        "keywords": "needs classification problem opportunity directive project objective",
        "length_intent": "Klasifikasikan kebutuhan klien ke dalam problem/opportunity/directive lalu tetapkan tujuan proyek dan jenis proyek yang paling tepat. (Target: 700 words)."
    },
    {
        "id": "c_4", "title": "BAB IV – PENDEKATAN",
        "subs": ["4.1 Acuan Prinsip/Framework/Teori/Regulasi", "4.2 Standar Penyelesaian Masalah"],
        "keywords": "approach framework principle theory regulation standard",
        "length_intent": "Uraikan acuan dan prinsip yang digunakan (framework/teori/regulasi/standar) sebagai landasan menyelesaikan masalah klien. (Target: 700 words)."
    },
    {
        "id": "c_5", "title": "BAB V – METODOLOGI",
        "subs": ["5.1 Alasan Pemilihan Metodologi", "5.2 Langkah Kerja dengan Framework Terpilih"],
        "keywords": "methodology rationale implementation steps framework",
        "length_intent": "Jelaskan mengapa metodologi tersebut dipilih dan bagaimana langkah kerja detail menggunakan framework tersebut. (Target: 800 words).",
        "visual_intent": "flowchart"
    },
    {
        "id": "c_6", "title": "BAB VI – SOLUTION DESIGN",
        "subs": ["6.1 Solusi/Output Metodologi yang Dibangun", "6.2 Kesesuaian Solusi terhadap Kebutuhan Klien"],
        "keywords": "solution design output deliverables target state",
        "length_intent": "Jelaskan desain solusi (output metodologi) yang akan dibangun atau diterapkan agar kebutuhan klien dapat tercapai. (Target: 800 words)."
    },
    {
        "id": "c_7", "title": "BAB VII – TIMELINE PEKERJAAN",
        "subs": ["7.1 Aktivitas per Fase", "7.2 Waktu Pelaksanaan dan Deliverable Tiap Fase"],
        "keywords": "timeline phase schedule milestone deliverable",
        "length_intent": "Jelaskan aktivitas yang dilakukan, kapan dilakukan, pada fase apa, serta deliverable pada setiap fase pekerjaan. (Target: 700 words).",
        "visual_intent": "gantt"
    },
    {
        "id": "c_8", "title": "BAB VIII – TATA KELOLA PROYEK",
        "subs": ["8.1 Mekanisme Pengambilan Keputusan", "8.2 Mekanisme Pengendalian Proyek"],
        "keywords": "project governance decision making controls monitoring",
        "length_intent": "Jelaskan tata kelola proyek, termasuk mekanisme pengambilan keputusan dan pengendalian agar proyek berjalan efektif. (Target: 700 words)."
    },
    {
        "id": "c_9", "title": "BAB IX – STRUKTUR & TEAM PROYEK",
        "subs": ["9.1 Struktur Tim Proyek", "9.2 Kapabilitas, Pengalaman, dan Sertifikasi"],
        "keywords": "project team structure capability experience certification",
        "length_intent": "Tunjukkan struktur dan komposisi tim proyek beserta kapabilitas, pengalaman, dan sertifikasi untuk membangun kepercayaan klien. (Target: 700 words)."
    },
    {
        "id": "c_10", "title": "BAB X – MODEL PEMBIAYAAN",
        "subs": ["10.1 Biaya dan Tahapan Pembayaran", "10.2 Model Pekerjaan dan Batasan Pekerjaan"],
        "keywords": "commercial pricing payment terms scope boundaries",
        "length_intent": "Jelaskan model bisnis proyek yang mencakup biaya, tahapan pembayaran, model pekerjaan, dan batasan pekerjaan secara jelas dan tegas. (Target: 700 words)."
    }
]

PERSONAS = {
    "c_1": "Senior Business Analyst",
    "c_2": "Principal Enterprise Architect",
    "c_3": "Lead Strategy Consultant",
    "c_4": "Lead Compliance & Governance Auditor",
    "c_5": "Senior Delivery Director",
    "c_6": "Chief Solution Architect",
    "c_7": "Master Project Manager",
    "c_8": "Program Governance Lead",
    "c_9": "Partner/Managing Director",
    "c_10": "Commercial Lead",
    "default": "Principal Management Consultant"
}

PROPOSAL_SYSTEM_PROMPT = """
You are an elite Principal Consultant and Technical Writer at {writer_firm}.
Your target audience is the Executive Board of {client}. Adopt this persona: {persona}.

--- STRICT WRITING RULES ---
1. NO FLUFF: Write strictly "singkat, padat, tidak bertele-tele" but still deep and specific (not generic).
2. FORMATTING (MANDATORY):
   - Use EXACT heading structure from provided H2 list.
   - Under each H2, create 2-3 H3 subsections (###) that are relevant and non-redundant.
   - Prioritize short explanatory paragraphs; use lists only when needed for sequence/checklist/comparison.
   - For numbered steps, keep sequence concise (typically 3-7 items) and avoid excessive nesting.
   - For bullets, every point must include context (not one-line fragments).
   - Include markdown tables only when content is naturally tabular (operational, governance, timeline, commercial).
3. DEPTH: Explain rationale, assumptions, risks, dependencies, metrics, and expected deliverables in actionable detail.
4. TONE: Professional, objective, direct, and highly persuasive in Indonesian. Do not use academic filler.
5. SYNTHESIS: Integrate OSINT data naturally as facts, do not mention "berdasarkan sumber online".

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

Mandatory output quality checks before finalizing:
- Provide concrete examples relevant to the client context.
- Ensure all key claims have business/technical rationale.
- Ensure the chapter is sufficiently detailed for board-level review and contributes proportionally to the total proposal target 20-25 pages (avoid over-expansion).

{visual_prompt}
DO NOT write greetings or introductions. Output strictly the chapter content.
"""
