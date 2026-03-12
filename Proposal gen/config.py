# config.py
import os


# SYSTEM MODE & API ADAPTER (HANDOVER SETTINGS)

DEMO_MODE = True 
FIRM_API_URL = "https://api.perusahaan-anda.com/v1" 
API_AUTH_TOKEN = "isi_token_disini_nanti"

# --- CREDENTIALS & HOSTS ---
GOOGLE_API_KEY = "API_KEY"
GOOGLE_CX_ID = "CX_ID"
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
    "Diagnostic": {
        "methodology": "1. Discovery & Data Gathering\n2. As-Is Architecture Analysis\n3. Gap Analysis & Risk Assessment\n4. Diagnostic Report Generation.",
        "team": "1x Principal Consultant, 2x Senior Business Analyst.",
        "commercial": "Fixed-fee based on scope. 50% Upfront, 50% upon Diagnostic Report delivery."
    },
    "Strategic": {
        "methodology": "1. Executive Alignment & Visioning\n2. To-Be Visioning & Target Operating Model\n3. Strategic Roadmap Development\n4. Board-Level Presentation.",
        "team": "1x Project Director, 1x Enterprise Architect, 1x Financial Modeler.",
        "commercial": "Retainer or Fixed-fee. Milestone payments tied to strategic sign-offs."
    },
    "Transformation": {
        "methodology": "1. Change Readiness Assessment\n2. Phased Rollout (Agile/Scrum)\n3. Organizational Change Management (OCM)\n4. Hypercare & Value Realization.",
        "team": "1x Program Manager, 2x Change Management Experts, 3x Technical/Domain Leads.",
        "commercial": "Time & Materials (T&M) or milestone-based over 12-18 months. Excludes software licensing."
    },
    "Implementation": {
        "methodology": "1. Requirement Traceability\n2. System Design & Configuration\n3. UAT & Integration Testing\n4. Deployment & Go-Live\n5. Handover to BAU.",
        "team": "1x Project Manager, 1x Lead Engineer, 3x System Implementers, 1x QA Specialist.",
        "commercial": "Milestone-based (20% Kickoff, 40% UAT, 30% Go-Live, 10% Handover)."
    }
}


# PROPOSAL STRUCTURE (TARGET: 20-25 PAGES TOTAL)

UNIVERSAL_STRUCTURE = [
    {
        "id": "c_1", "title": "BAB I – KONTEKS ORGANISASI",
        "subs": ["1.1 Latar Belakang Perusahaan", "1.2 Dinamika dan Visi Saat Ini"],
        "keywords": "context background vision strategy",
        "length_intent": "Provide a thorough background using the OSINT data. Detail the market conditions and the macro-level vision. (Target: 500 words)."
    },
    {
        "id": "c_2", "title": "BAB II – PERMASALAHAN",
        "subs": ["2.1 Tantangan Utama dan Root Cause Analysis", "2.2 Dampak Terhadap Bisnis dan Risiko Masa Depan"],
        "keywords": "problem pain points bottleneck issue root cause",
        "length_intent": "Perform a focused Root Cause Analysis (RCA). Theorize why it is happening structurally and quantify the business risks. (Target: 600 words)."
    },
    {
        "id": "c_3", "title": "BAB III – SOLUSI DAN PENDEKATAN",
        "subs": ["3.1 Klasifikasi Kebutuhan Strategis", "3.2 Arsitektur Pendekatan Solusi"],
        "keywords": "solution approach diagnostic strategic implementation architecture",
        "length_intent": "Deliver a technical explanation of the proposed solution. Break down how the solution resolves the Root Causes. (Target: 600 words)."
    },
    {
        "id": "c_4", "title": "BAB IV – POTENSI KERANGKA KERJA",
        "subs": ["4.1 Pemetaan Framework Global", "4.2 Pemenuhan Kepatuhan dan Regulasi Nasional"],
        "keywords": "framework iso cobit itil regulation compliance",
        "length_intent": "Provide an authoritative breakdown of EVERY requested framework based on the OSINT data. Explain practical implementation. (Target: 500 words)."
    },
    {
        "id": "c_5", "title": "BAB V – MANFAAT",
        "subs": ["5.1 Efisiensi dan Optimalisasi Operasional", "5.2 ROI, Skalabilitas, dan Nilai Tambah Bisnis"],
        "keywords": "benefit roi advantage growth efficiency scalability",
        "length_intent": "Write a strong business case. Include hypothetical percentages and long-term strategic advantages. (Target: 400 words)."
    },
    {
        "id": "c_6", "title": "BAB VI – METODOLOGI",
        "subs": ["6.1 Rincian Pendekatan Pelaksanaan Berfase", "6.2 Manajemen Risiko dan Jaminan Kualitas"],
        "keywords": "methodology quality assurance process framework risk mitigation",
        "length_intent": "Explain the inputs, processes, and expected outputs of every phase. Detail how risks are mitigated. (Target: 600 words).",
        "visual_intent": "flowchart"
    },
    {
        "id": "c_7", "title": "BAB VII – TAHAPAN DAN WAKTU",
        "subs": ["7.1 Penjabaran Jadwal Pelaksanaan", "7.2 Milestone Kritis dan Kriteria Penerimaan"],
        "keywords": "timeline phase schedule milestone sprint",
        "length_intent": "Provide a narrative describing what happens in the timeline phases. Define Acceptance Criteria. (Target: 400 words).",
        "visual_intent": "gantt"
    },
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

PERSONAS = {
    "c_1": "Senior Business Analyst",
    "c_2": "Principal Enterprise Architect",
    "c_3": "Chief Technology Officer",
    "c_4": "Lead Compliance & Governance Auditor",
    "c_5": "Chief Financial Officer",
    "c_6": "Senior Delivery Director",
    "c_7": "Master Project Manager",
    "c_8": "Partner/Managing Director",
    "c_9": "Commercial Lead",
    "c_10": "Senior Account Executive",
    "default": "Principal Management Consultant"
}

PROPOSAL_SYSTEM_PROMPT = """
You are an elite Principal Consultant and Technical Writer at {writer_firm}.
Your target audience is the Executive Board of {client}. Adopt this persona: {persona}.

--- SYSTEM CONTEXT ---
Global Client Data: {global_data}
Latest Client News: {client_news}
Regulatory/Framework Context: {regulation_data}
Writer Firm Contact Info: {writer_data}
Writer Firm Portfolio/Experience: {firm_exp_data}
Writer Firm Collaboration with Client: {collab_data}
Historical Vector Data: {structured_row_data}
Semantic DB Matches: {rag_data}

--- DYNAMIC INSTRUCTIONS (FROM UI INPUTS) ---
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

Rules:
- Tone MUST be professional, objective, academic, and highly persuasive in Indonesian.
- {visual_prompt}
- DO NOT write an introduction or greeting. Go straight into the content.
"""