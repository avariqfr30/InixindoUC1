"""
Configuration and settings module for the Strategic Proposal Engine.
Contains API keys, model definitions, firm mock data, and prompt templates.
"""

from typing import Dict, List, Tuple, Any

# =====================================================================
# SYSTEM & API SETTINGS
# =====================================================================
DEMO_MODE: bool = True 
FIRM_API_URL: str = "https://api.perusahaan-anda.com/v1" 
API_AUTH_TOKEN: str = "isi_token_disini_nanti"

SERPER_API_KEY: str = "YOUR_SERPER_API_KEY"
OLLAMA_HOST: str = "http://127.0.0.1:11434"

LLM_MODEL: str = "gpt-oss:120b-cloud" 
EMBED_MODEL: str = "bge-m3:latest"
DB_URI: str = "sqlite:///projects.db" 

# =====================================================================
# FIRM BRANDING & IDENTITY
# =====================================================================
WRITER_FIRM_NAME: str = "Inixindo Jogja" 
DEFAULT_COLOR: Tuple[int, int, int] = (0, 51, 102)

DATA_MAPPING: Dict[str, str] = {
    "entity": "Client Entity",
    "topic": "Strategic Initiative",
    "budget": "Investment Estimation"
}

# =====================================================================
# INTERNAL DATABASE MOCK
# =====================================================================
MOCK_FIRM_PROFILE: Dict[str, str] = {
    "contact_info": "Kantor Pusat Inixindo Jogja\nJl. Kenari No. 69, Yogyakarta 55165\nEmail: info@inixindo.id\nTelp: (0274) 515448",
    "portfolio_highlights": "Pengalaman 30+ tahun dalam Transformasi TI, Audit, IT Master Plan, dan DevSecOps untuk Tier-1 Enterprise."
}

MOCK_FIRM_STANDARDS: Dict[str, Dict[str, str]] = {
    "Diagnostic": {
        "methodology": "1. Discovery\n2. As-Is Analysis\n3. Gap Analysis\n4. Diagnostic Report.",
        "team": "1x Principal Consultant, 2x Senior Business Analyst.",
        "commercial": "Fixed-fee. 50% Upfront, 50% on Delivery."
    },
    "Strategic": {
        "methodology": "1. Executive Alignment\n2. To-Be Visioning\n3. Strategic Roadmap.",
        "team": "1x Project Director, 1x Enterprise Architect.",
        "commercial": "Retainer or Fixed-fee based on milestones."
    },
    "Transformation": {
        "methodology": "1. Readiness Assessment\n2. Agile Rollout\n3. Change Management\n4. Hypercare.",
        "team": "1x Program Manager, 3x Domain Leads.",
        "commercial": "Time & Materials (T&M) 12-18 months."
    },
    "Implementation": {
        "methodology": "1. System Design\n2. Integration Testing\n3. Deployment\n4. Handover.",
        "team": "1x Project Manager, 1x Lead Engineer, 1x QA.",
        "commercial": "Milestone-based (20% Kickoff, 40% UAT, 30% Go-Live, 10% Handover)."
    }
}

# =====================================================================
# PROPOSAL ARCHITECTURE (CONCISE & EXECUTIVE FOCUSED)
# =====================================================================
UNIVERSAL_STRUCTURE: List[Dict[str, Any]] = [
    {
        "id": "c_1", "title": "BAB I – KONTEKS ORGANISASI",
        "subs": ["1.1 Latar Belakang Klien", "1.2 Urgensi Inisiatif"],
        "keywords": "context background vision strategy",
        "length_intent": "Singkat, padat, eksekutif (Maks 150 kata). Fokus pada urgensi bisnis berdasarkan data OSINT."
    },
    {
        "id": "c_2", "title": "BAB II – PERMASALAHAN",
        "subs": ["2.1 Tantangan Utama", "2.2 Analisis Akar Masalah (Root Cause)"],
        "keywords": "problem pain points bottleneck issue root cause",
        "length_intent": "Gunakan bullet points. Hindari teori. Langsung tunjukkan letak bottleneck dan risiko bisnis nyata."
    },
    {
        "id": "c_3", "title": "BAB III – SOLUSI DAN PENDEKATAN",
        "subs": ["3.1 Arsitektur Solusi", "3.2 Pemenuhan Kebutuhan Strategis"],
        "keywords": "solution approach strategic implementation architecture",
        "length_intent": "To the point. Jelaskan spesifikasi solusi teknis untuk menyelesaikan masalah secara logis."
    },
    {
        "id": "c_4", "title": "BAB IV – KEPATUHAN & REGULASI",
        "subs": ["4.1 Pemenuhan Regulasi Global", "4.2 Standar Kepatuhan Nasional"],
        "keywords": "framework iso cobit itil regulation compliance",
        "length_intent": "Ringkas. Sebutkan framework yang relevan dan apa output kepatuhannya bagi klien. Gunakan list."
    },
    {
        "id": "c_5", "title": "BAB V – MANFAAT & ROI",
        "subs": ["5.1 Optimalisasi Operasional", "5.2 Nilai Tambah Bisnis (ROI)"],
        "keywords": "benefit roi advantage efficiency",
        "length_intent": "Pemaparan eksekutif berfokus pada efisiensi angka, metrik pertumbuhan, dan penghematan biaya."
    },
    {
        "id": "c_6", "title": "BAB VI – METODOLOGI",
        "subs": ["6.1 Fase Pelaksanaan", "6.2 Penjaminan Mutu"],
        "keywords": "methodology quality assurance process risk",
        "length_intent": "Sangat terstruktur. Gunakan penomoran untuk tahapan (phases). Dilarang menjelaskan teori framework dasar.",
    },
    {
        "id": "c_7", "title": "BAB VII – JADWAL & MILESTONE",
        "subs": ["7.1 Garis Waktu Eksekusi", "7.2 Kriteria Penerimaan (Acceptance Criteria)"],
        "keywords": "timeline phase schedule milestone sprint",
        "length_intent": "Singkat. Jabarkan pembagian waktu kerja secara pragmatis.",
    },
    {
        "id": "c_8", "title": "BAB VIII – KAPABILITAS TIM",
        "subs": ["8.1 Tata Kelola Proyek", "8.2 Rekam Jejak Konsultan"],
        "keywords": "team expert role portfolio",
        "length_intent": "Fokus pada struktur tim yang efisien dan sintesis portofolio nyata dari perusahaan konsultan."
    },
    {
        "id": "c_9", "title": "BAB IX – ESTIMASI BIAYA",
        "subs": ["9.1 Rekomendasi Investasi Berjenjang", "9.2 Ketentuan Komersial"],
        "keywords": "budget cost commercial investment decoy",
        "length_intent": """CRITICAL - FINANCIAL-BACKED DECOY PRICING:
Buatlah tabel Markdown berisi 3 opsi harga (Esensial, Rekomendasi, Premium).
Opsi harus disesuaikan secara logis dengan analisa kekuatan finansial / skala bisnis klien.
Opsi 'Rekomendasi' harus terlihat sebagai pilihan paling rasional dan sepadan dengan ROI."""
    },
    {
        "id": "c_10", "title": "BAB X – PENUTUP",
        "subs": ["10.1 Kesimpulan Eksekutif", "10.2 Tindak Lanjut"],
        "keywords": "closing commitment next steps",
        "length_intent": "Satu paragraf penutup yang tajam (Call to Action). Akhiri dengan detail kontak."
    }
]

# =====================================================================
# LLM SYSTEM PROMPT
# =====================================================================
PROPOSAL_SYSTEM_PROMPT: str = """
You are an elite Principal Consultant and Technical Writer at {writer_firm}.
Client: {client}. Project Initiative: {project_status} - {project_type}.

--- EXECUTIVE COMMUNICATION RULES (ANTI-BERTELE-TELE) ---
1. SINGKAT & PADAT: Executives do not read long paragraphs. Keep sentences short.
2. NO THEORETICAL FLUFF: Do not explain basic concepts. Explain the IMPACT instead.
3. BULLET POINTS & BOLDING: Heavily use bullet points and bold text for scanning readability.
4. PARALLEL STRUCTURE: Every bullet point must be grammatically parallel.

--- CLIENT CONTEXT (MARCH 2026 INTELLIGENCE) ---
Global Client Data: {global_data}
Latest Client News: {client_news}
Client Financial Strength (Revenue/Funding): {client_financials}
Regulatory/Framework Context: {regulation_data}
Client Collaboration History: {collab_data}

--- DYNAMIC INSTRUCTIONS ---
{extra_instructions}

--- TASK ---
Write "{chapter_title}".
Include these sub-chapters exactly as H2 (##):
{sub_chapters}

Follow the length and formatting intent strictly: {length_intent}
"""