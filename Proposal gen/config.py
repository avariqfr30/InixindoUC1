# config.py
import os


# SYSTEM MODE & API ADAPTER (HANDOVER SETTINGS)

DEMO_MODE = True 
FIRM_API_URL = "https://api.perusahaan-anda.com/v1" 
API_AUTH_TOKEN = "isi_token_disini_nanti"

# --- CREDENTIALS & HOSTS ---
SERPER_API_KEY = "YOUR_SERPER_API_KEY" # Serper.dev Key
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


# KNOWLEDGE PATTERN LIBRARY (Intelligent Autocomplete di UI)

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
        "regulations": ["Permenkes No 24 Tahun 2022 (Rekam Medis Elektronik)", "HIPAA", "ISO 27001"]
    },
    "general_it": {
        "keywords": ["infrastruktur", "cloud", "aplikasi", "software", "jaringan", "data center"],
        "regulations": ["ITIL v4", "COBIT 2019", "TOGAF", "ISO 20000", "ISO 27001"]
    }
}


# MOCK API DATA (Internal Firm Standards)

MOCK_FIRM_STANDARDS = {
    "Diagnostic": {"methodology": "1. Discovery\n2. As-Is Analysis\n3. Gap Analysis\n4. Report", "team": "1x Principal Consultant, 1x Business Analyst.", "commercial": "Fixed-fee based on scope."},
    "Strategic": {"methodology": "1. Executive Alignment\n2. To-Be Visioning\n3. Roadmap", "team": "1x Project Director, 1x Enterprise Architect.", "commercial": "Retainer or Fixed-fee."},
    "Transformation": {"methodology": "1. Readiness Assessment\n2. Phased Rollout\n3. OCM\n4. Hypercare", "team": "1x Program Manager, 2x Domain Leads.", "commercial": "Time & Materials (T&M)."},
    "Implementation": {"methodology": "1. Design\n2. UAT & Testing\n3. Deployment\n4. Go-Live", "team": "1x Project Manager, 1x Lead Engineer.", "commercial": "Milestone-based (20% Kickoff, 40% UAT, 30% Go-Live, 10% Handover)."}
}


# PROPOSAL STRUCTURE (TARGET: 20-25 PAGES TOTAL)

UNIVERSAL_STRUCTURE = [
    # --- PHASE 1: PEMAHAMAN DASAR ---
    {
        "id": "c_1", "title": "BAB I – KONTEKS KLIEN",
        "subs": ["1.1 Latar Belakang Organisasi", "1.2 Alasan Permintaan Konsultasi"],
        "keywords": "context background vision strategy",
        "length_intent": "Sharp and concise (max 300 words). Memahami latar belakang organisasi klien yang menjadi dasar/alasan meminta jasa konsultasi."
    },
    {
        "id": "c_2", "title": "BAB II – PERMASALAHAN",
        "subs": ["2.1 Identifikasi Tantangan Utama", "2.2 Ekspektasi dan Kebutuhan Aktual"],
        "keywords": "problem pain points bottleneck issue",
        "length_intent": "Use BULLET POINTS. Menangkap apa yang benar-benar menjadi kebutuhan atau keinginan klien. Jangan bertele-tele."
    },
    {
        "id": "c_3", "title": "BAB III – KLASIFIKASI KEBUTUHAN",
        "subs": ["3.1 Klasifikasi (Problem/Opportunity/Directive)", "3.2 Tujuan dan Jenis Proyek"],
        "keywords": "classification goals objective opportunity directive",
        "length_intent": "Mengklasifikasikan kebutuhan dalam problem/opportunity/directive dan menemukan tujuan serta jenis proyek. Gunakan metrik/angka tebal (bold)."
    },

    # --- PHASE 2: METODOLOGI & SOLUSI ---
    {
        "id": "c_4", "title": "BAB IV – PENDEKATAN",
        "subs": ["4.1 Acuan Framework dan Teori", "4.2 Standar Kepatuhan dan Regulasi"],
        "keywords": "framework iso cobit itil regulation compliance approach",
        "length_intent": "Menentukan acuan/prinsip yang digunakan (framework/teori/regulasi/standar) untuk menyelesaikan masalah yang ada. Gunakan list."
    },
    {
        "id": "c_5", "title": "BAB V – METODOLOGI",
        "subs": ["5.1 Alasan Pemilihan Pendekatan", "5.2 Langkah Kerja Berbasis Framework"],
        "keywords": "methodology process steps",
        "length_intent": "Mengapa memilih pendekatan tersebut? Dan bagaimana langkah kerja menggunakan framework tersebut? Gunakan penomoran (numbered lists) yang terstruktur."
    },
    {
        "id": "c_6", "title": "BAB VI – SOLUTION DESIGN",
        "subs": ["6.1 Desain Solusi Utama", "6.2 Pencapaian Kebutuhan Klien"],
        "keywords": "solution design architecture output",
        "length_intent": "Menjelaskan solusi (output metodologi) apa yang akan dibangun atau diterapkan, agar kebutuhan klien dapat tercapai. Detail namun tajam."
    },

    # --- PHASE 3: EKSEKUSI ---
    {
        "id": "c_7", "title": "BAB VII – TIMELINE PEKERJAAN",
        "subs": ["7.1 Fase dan Aktivitas Pekerjaan", "7.2 Deliverable Tiap Fase"],
        "keywords": "timeline phase schedule deliverable",
        "length_intent": "Menjelaskan apa yang dilakukan, kapan dilakukan, dan dalam fase apa aktivitas tersebut terjadi serta deliverable tiap fase nya."
    },
    {
        "id": "c_8", "title": "BAB VIII – TATA KELOLA PROYEK",
        "subs": ["8.1 Mekanisme Pengambilan Keputusan", "8.2 Pengendalian dan Penjaminan Mutu"],
        "keywords": "governance quality control decision making",
        "length_intent": "Menjelaskan mekanisme pengambilan Keputusan dan pengendalian yang digunakan untuk memastikan proyek berjalan."
    },
    {
        "id": "c_9", "title": "BAB IX – STRUKTUR & TEAM PROYEK",
        "subs": ["9.1 Susunan Tim Eksekusi", "9.2 Kapabilitas, Pengalaman, dan Sertifikasi"],
        "keywords": "team expert role structure portfolio capability",
        "length_intent": "Menunjukkan kapabilitas konsultan untuk membangun kepercayaan klien (struktur tim, pengalaman dan sertifikasi). Sintesis data portofolio dengan percaya diri."
    },

    # --- PHASE 4: KOMERSIAL ---
    {
        "id": "c_8", "title": "BAB VIII – TIM DAN KAPABILITAS",
        "subs": ["8.1 Struktur Organisasi Tata Kelola Proyek", "8.2 Profil Peran dan Rekam Jejak Firm"],
        "keywords": "team expert role structure director manager portfolio",
        "length_intent": "Detail the governance structure. Base the firm's historical excellence entirely on the retrieved OSINT data. (Target: 500 words)."
    },
    {
        "id": "c_9", "title": "BAB IX – ESTIMASI BIAYA",
        "subs": ["9.1 Rincian Investasi Komprehensif", "9.2 Ketentuan Komersial dan Asumsi Legal"],
        "keywords": "budget cost commercial investment terms assumption legal",
        "length_intent": "Provide a Markdown table for costs. Detail the assumptions, exclusions, and commercial terms. (Target: 400 words)."
    },
    {
        "id": "c_10", "title": "BAB X – PENUTUP",
        "subs": ["10.1 Kesimpulan Eksekutif", "10.2 Rencana Tindak Lanjut (Next Steps)"],
        "keywords": "closing commitment next steps",
        "length_intent": "Write a strong executive conclusion. End with a call to action and the firm's contact details sourced from OSINT. (Target: 200 words)."
    }
]

PROPOSAL_SYSTEM_PROMPT = """
You are an elite Proposal Architect for {writer_firm}.
Client: {client} | Industry: {industry} | Employees: {employee_count}
Project Type: {project_status} - {project_type}

--- SYSTEM CONTEXT ---
Global Client Data: {global_data}
Latest Client News: {client_news}
Regulatory/Framework Context: {regulation_data}
Writer Firm Contact Info: {writer_data}
Writer Firm Portfolio/Experience: {firm_exp_data}
Writer Firm Collaboration with Client: {collab_data}
Historical Vector Data: {structured_row_data}
Semantic DB Matches: {rag_data}

--- HIGH-IMPACT STRUCTURE RULE ---
1. Be comprehensive but highly structured. Executives appreciate depth but hate unstructured text walls.
2. Ensure you write thoroughly (target 300-400 words per chapter) to build a persuasive business case.
3. Use bold text to highlight key numbers or outcomes.

--- STRICT TYPOGRAPHY & LIST FORMATTING RULES ---
1. UNORDERED LISTS: You MUST use exactly the '-' (hyphen) character followed by a space. You are strictly forbidden from using '*', '•', '➢', or any other bullet character.
2. ORDERED LISTS: You MUST use standard numerical format ('1.', '2.', '3.'). Do not use alphabetical ('a)', 'b)') or roman numerals ('i.', 'ii.').
3. LIST PARALLELISM: Every bullet point MUST start with a Capital letter and end with a period (.).
4. SPACING: You MUST place a blank line before and after every list to ensure the Markdown parser renders it correctly.
5. NO DEEP NESTING: Maximum one level of indentation. Do not create deeply nested, confusing sub-lists.

--- DEEP PERSONALIZATION & FORMATTING RULES ---
1. NO H1 TITLES: Do NOT output the main Chapter Title (e.g., # BAB I). Start your response directly with the H2 (##) sub-chapters.
2. NAME DROPPING: NEVER use generic terms like "Klien", "Perusahaan Anda", atau "Organisasi". You MUST explicitly use the exact client name: "{client}" throughout the text.
3. WEAVE CONTEXT: Intertwine their specific industry ({industry}) and their real-world news ({client_news}) naturally into your arguments.

--- OSINT & API CONTEXT ---
Client News: {client_news}
Firm API Portfolio: {firm_api_portfolio}
Firm API Contact: {firm_api_contact}

--- DYNAMIC UI INSTRUCTIONS ---
Target Scope: {scope}
Target Outcome/Goals: {outcome}
Timeline: {timeline} | Target Budget: {budget}
Extra Notes: {notes}
{extra_instructions}

--- LENGTH DIRECTIVE ---
You are writing a professional consulting proposal. Ensure high technical detail and deep analytical thought, but remain structured.
Read the length intent carefully: {length_intent}

--- STRICT ACCURACY & RAW DATA SYNTHESIS RULE ---
1. You must use the 'Writer Firm Contact Info' and 'Writer Firm Portfolio/Experience' provided in the OSINT system context. 
2. CRITICAL: The OSINT data is provided as raw Google snippets containing fragmented sentences and ellipses (...). DO NOT output these fragments raw. You MUST synthesize these fragmented facts into smooth, professional, and highly confident sentences.
3. If the OSINT data for the firm is sparse or unclear, confidently extrapolate using general enterprise best practices. NEVER state or admit that your firm lacks experience, lacks a public portfolio, or that information is unavailable. Frame all capabilities with supreme confidence.

--- TASK ---
Write "{chapter_title}".
Include these sub-chapters exactly as H2 (##):
{sub_chapters}
Follow the length intent: {length_intent}
"""