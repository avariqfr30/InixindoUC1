# config.py
import os

# =====================================================================
# SYSTEM MODE & API ADAPTER (HANDOVER SETTINGS)
# =====================================================================
# Ubah ke False saat Handover ke tim IT Perusahaan agar sistem membaca API asli
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

# =====================================================================
# KNOWLEDGE PATTERN LIBRARY (Intelligent Autocomplete di UI)
# =====================================================================
SMART_SUGGESTIONS = {
    "banking": {
        "keywords": ["bank", "bca", "bri", "mandiri", "finance", "fintech", "payment", "kredit"],
        "regulations": ["POJK (Otoritas Jasa Keuangan)", "PBI (Bank Indonesia)", "ISO 27001 (Keamanan Informasi)", "UU PDP"],
        "pain_points": ["Ketidakpatuhan regulasi OJK/BI", "Ancaman siber pada transaksi digital", "Persaingan ketat dengan bank digital", "Sistem core banking legacy"]
    },
    "government": {
        "keywords": ["kementerian", "dinas", "pemerintah", "badan", "provinsi", "kabupaten", "gov"],
        "regulations": ["Perpres SPBE", "Satu Data Indonesia", "UU KIP", "Peraturan BSSN"],
        "pain_points": ["Silo data antar instansi", "Proses birokrasi masih manual", "Infrastruktur IT belum terpusat", "Keterbatasan kapabilitas digital ASN"]
    },
    "security": {
        "keywords": ["security", "cyber", "ransomware", "siber", "keamanan", "soc", "penetration"],
        "regulations": ["ISO 27001:2022", "NIST Cybersecurity Framework", "BSSN Guidelines", "CIS Controls"],
        "pain_points": ["Tidak ada visibilitas aset IT", "Rentan terhadap serangan Ransomware", "Kurangnya kesadaran keamanan (Security Awareness)", "Belum ada SOP penanganan insiden"]
    }
}

# =====================================================================
# MOCK API: PROPRIETARY FIRM STANDARDS (Aktif saat DEMO_MODE = True)
# =====================================================================
MOCK_FIRM_STANDARDS = {
    "Diagnostic": {
        "methodology": "Inixindo Diagnostic Framework (Phase 1: Initiate, Phase 2: Assess, Phase 3: Gap Analysis, Phase 4: Roadmap)",
        "team": "Project Director, Lead Assessor, Subject Matter Expert (SME), Business Analyst",
        "commercial": "Fixed Price. 50% Downpayment, 50% Final Report. Scope limited to assessment only."
    },
    "Strategic": {
        "methodology": "Inixindo Strategic Blueprint (Phase 1: Current State Analysis, Phase 2: Target Operating Model, Phase 3: Strategic Roadmap, Phase 4: Executive Alignment)",
        "team": "Managing Consultant, Enterprise Architect, Domain Specialist",
        "commercial": "Fixed Price based on Deliverables. 30% Kick-off, 40% Draft Blueprint, 30% Final Sign-off."
    },
    "Implementation": {
        "methodology": "Inixindo Agile Deployment (Phase 1: Solution Design, Phase 2: Build & Configure, Phase 3: UAT, Phase 4: Go-Live)",
        "team": "Project Manager, Technical Architect, System Engineers, QA Tester",
        "commercial": "Milestone Based: 30% Kick-off, 40% UAT Sign-off, 30% Go-Live. Includes 1-month warranty."
    },
    "Transformation": {
        "methodology": "Inixindo Capability-Driven Transformation (Phase 1: Envisioning, Phase 2: Capability Build, Phase 3: Change Management, Phase 4: Value Realization)",
        "team": "Project Director, Change Management Lead, Technical Architect, HR/Culture Consultant",
        "commercial": "Time & Material or Retainer. Billed monthly based on mandays consumed and achieved transformation KPIs."
    }
}

# =====================================================================
# CORE SYSTEM PROMPT
# =====================================================================
PROPOSAL_SYSTEM_PROMPT = """
You are a Senior Consultant for {client} working at {writer_firm}. 
ROLE: {persona}.

=== LIVE EXTERNAL OSINT (SEARCH API) ===
Client News & Context: {client_news}
Regulatory Mandates: {regulation_data}
Client Profile: {global_data}

=== EXACT PROJECT REQUIREMENTS (UI INPUT) ===
{structured_row_data}
Project Goal: {project_goal}
Project Type: {project_type}
Target Timeline: {timeline}
Discovery Notes: {discovery_notes}

=== PROPRIETARY FIRM STANDARDS (INTERNAL API) ===
You MUST adhere to these firm standards. Do not invent your own.
- Firm Methodology: {firm_methodology}
- Required Team Structure: {firm_team}
- Commercial Rules: {firm_commercial}

=== HISTORICAL VECTOR CONTEXT ===
{rag_data}

MANDATORY RULES:
1. STRICT LANGUAGE ENFORCEMENT: YOU MUST WRITE THE ENTIRE RESPONSE STRICTLY IN BAHASA INDONESIA.
2. DO NOT repeat the Chapter Title in your output.
3. HIGH INFORMATION DENSITY. Write comprehensively but concisely. 
4. TARGET LENGTH & STYLE: {length_intent}
5. Ground your response ONLY in the provided contexts. Do not hallucinate capabilities.
6. {visual_prompt}
7. {extra_instructions}

WRITE CONTENT FOR '{chapter_title}' covering:
{sub_chapters}
"""

# ==========================================
# --- SCHEMA 1: TRAINING STRUCTURE (9 BAB) ---
# ==========================================
TRAINING_STRUCTURE = [
    {
        "id": "chap_1", "title": "BAB I – LATAR BELAKANG",
        "subs": ["1.1 Gambaran Umum Tantangan Organisasi", "1.2 Perubahan Lingkungan Bisnis & Teknologi", "1.3 Kesenjangan Kompetensi / Kematangan Digital", "1.4 Dampak Jika Masalah Tidak Ditangani", "1.5 Urgensi Program Pelatihan &/atau Konsultasi"],
        "keywords": "problem pain points business environment gap urgency",
        "visual_intent": "bar_chart",
        "length_intent": "Detailed and comprehensive. (Target: 400-500 words)."
    },
    {
        "id": "chap_2", "title": "BAB II – MAKSUD DAN TUJUAN PROGRAM",
        "subs": ["2.1 Maksud Program", "2.2 Tujuan Umum", "2.3 Tujuan Khusus", "2.4 Sasaran Program", "2.5 Ruang Lingkup Program"],
        "keywords": "objectives scope targets",
        "length_intent": "Structured and descriptive. (Target: 250-350 words)."
    },
    {
        "id": "chap_3", "title": "BAB III – SOLUSI YANG DITAWARKAN",
        "subs": ["3.1 Pendekatan Solusi", "3.2 Kerangka Solusi Berbasis Kebutuhan Klien", "3.3 Posisi Solusi dalam Peta Transformasi", "3.4 Nilai Tambah & Diferensiasi", "3.5 Solusi Pelatihan: Konsep, Learning Path, Metodologi, Metode Pelaksanaan, Output"],
        "keywords": "solution framework training methodology",
        "visual_intent": "flowchart",
        "length_intent": "Highly detailed, technical, and methodical. (Target: 600 words)."
    },
    {
        "id": "chap_4", "title": "BAB IV – RENCANA AKSI & IMPLEMENTASI",
        "subs": ["4.1 Tahapan Pelaksanaan Program", "4.2 Timeline / Jadwal Kegiatan", "4.3 Peran dan Tanggung Jawab (RACI)", "4.4 Mekanisme Koordinasi & Pelaporan", "4.5 Manajemen Risiko"],
        "keywords": "action plan timeline raci risk management",
        "visual_intent": "gantt",
        "length_intent": "Structured and descriptive. Use Markdown tables. (Target: 400 words)."
    },
    {
        "id": "chap_5", "title": "BAB V – OUTPUT, OUTCOME, DAN INDIKATOR KEBERHASILAN",
        "subs": ["5.1 Output Program", "5.2 Outcome Jangka Pendek", "5.3 Outcome Jangka Menengah & Panjang", "5.4 KPI Keberhasilan", "5.5 Mekanisme Evaluasi"],
        "keywords": "kpi output outcome evaluation",
        "length_intent": "Impact-focused. Explain the metrics cleanly. (Target: 300 words)."
    },
    {
        "id": "chap_6", "title": "BAB VI – TIM PELAKSANA",
        "subs": ["6.1 Struktur Tim", "6.2 Peran & Kompetensi Tim", "6.3 Pengalaman Relevan Tim"],
        "keywords": "team structure roles competence",
        "length_intent": "Professional profiles and team hierarchy. (Target: 200 words)."
    },
    {
        "id": "chap_7", "title": "BAB VII – PORTOFOLIO & PENGALAMAN",
        "subs": ["7.1 Profil Perusahaan", "7.2 Portofolio Pelatihan", "7.3 Klien & Mitra", "7.4 Testimoni"],
        "keywords": "portfolio profile credibility Inixindo",
        "length_intent": "Persuasive and credential-focused. (Target: 250 words)."
    },
    {
        "id": "chap_8", "title": "BAB VIII – SKEMA BIAYA & INVESTASI",
        "subs": ["8.1 Ruang Lingkup Biaya", "8.2 Skema Pembiayaan", "8.3 Value vs Investment", "8.4 Ketentuan Pembayaran"],
        "keywords": "investment cost pricing ROI",
        "length_intent": "Clear and detailed. Use Markdown tables for costs. (Target: 200 words)."
    },
    {
        "id": "chap_9", "title": "BAB IX – PENUTUP",
        "subs": ["9.1 Komitmen", "9.2 Harapan Kerja Sama", "9.3 Kontak & Tindak Lanjut"],
        "keywords": "closing contact commitment",
        "length_intent": "Professional and impactful closing statement. (Target: 100 words)."
    }
]

# ==============================================
# --- SCHEMA 2: CONSULTING STRUCTURE (12 BAB) ---
# ==============================================
CONSULTING_STRUCTURE = [
    {
        "id": "c_chap_1", "title": "BAB I – EXECUTIVE SUMMARY",
        "subs": ["1.1 Ringkasan Konteks Organisasi Klien", "1.2 Ringkasan Masalah Utama", "1.3 Ringkasan Solusi yang Ditawarkan", "1.4 Dampak atau Manfaat bagi Organisasi"],
        "keywords": "executive summary context problem solution impact",
        "length_intent": "Highly concise executive summary. High impact, persuasive tone. (Target: 250 words)."
    },
    {
        "id": "c_chap_2", "title": "BAB II – KONTEKS KLIEN (CLIENT CONTEXT)",
        "subs": ["2.1 Profil Organisasi", "2.2 Sektor Industri", "2.3 Kondisi Bisnis Saat Ini", "2.4 Sistem atau Kapabilitas yang Telah Dimiliki", "2.5 Tantangan Organisasi yang Relevan"],
        "keywords": "profile industry business condition capabilities challenges",
        "visual_intent": "bar_chart",
        "length_intent": "Detailed and heavily researched. Use bullet points for clear reading. (Target: 400 words)."
    },
    {
        "id": "c_chap_3", "title": "BAB III – PERMASALAHAN (PROBLEM STATEMENT)",
        "subs": ["3.1 Current Situation", "3.2 Key Challenge", "3.3 Impact atau Risk Terhadap Organisasi", "3.4 Need for Solution"],
        "keywords": "current situation key challenge impact risk need for solution",
        "length_intent": "Direct and analytical. Clearly define the pain points. (Target: 300 words)."
    },
    {
        "id": "c_chap_4", "title": "BAB IV – KLASIFIKASI KEBUTUHAN (CLIENT NEED CLASSIFICATION)",
        "subs": ["4.1 Identifikasi Jenis Kebutuhan (Problem / Opportunity / Directive)", "4.2 Jenis Proyek (Diagnostic / Strategic / Implementation / Transformation)"],
        "keywords": "problem opportunity directive diagnostic strategic transformation",
        "length_intent": "Categorical and structured. Define the exact classification of the project. (Target: 200 words)."
    },
    {
        "id": "c_chap_5", "title": "BAB V – PENDEKATAN KONSULTANSI (CONSULTING APPROACH)",
        "subs": ["5.1 Prinsip Pendekatan", "5.2 Standar atau Framework yang Digunakan"],
        "keywords": "risk-based governance capability framework standard approach",
        "length_intent": "Methodical and authoritative. Explain the overarching philosophy of the solution. (Target: 350 words)."
    },
    {
        "id": "c_chap_6", "title": "BAB VI – METODOLOGI PROYEK",
        "subs": ["6.1 Phase 1 – Assessment", "6.2 Phase 2 – Design", "6.3 Phase 3 – Implementation", "6.4 Phase 4 – Evaluation"],
        "keywords": "assessment design implementation evaluation methodology",
        "visual_intent": "flowchart",
        "length_intent": "Step-by-step breakdown. Very structured using bullet points for phases. (Target: 400 words)."
    },
    {
        "id": "c_chap_7", "title": "BAB VII – SOLUTION DESIGN",
        "subs": ["7.1 Target State Organisasi", "7.2 Governance Model", "7.3 Process Framework", "7.4 Technology Architecture", "7.5 Capability Development", "7.6 Deliverable Utama"],
        "keywords": "target state governance model process framework technology capability deliverable",
        "length_intent": "The core technical proposal. Highly detailed, methodical, and deep. (Target: 600 words)."
    },
    {
        "id": "c_chap_8", "title": "BAB VIII – TIMELINE IMPLEMENTASI",
        "subs": ["8.1 Rencana Waktu Pelaksanaan Tiap Fase", "8.2 Milestone Penting"],
        "keywords": "timeline phase milestone assessment design implementation",
        "visual_intent": "gantt",
        "length_intent": "Strictly structured. Rely on Markdown tables for the Timeline and milestones. (Target: 250 words)."
    },
    {
        "id": "c_chap_9", "title": "BAB IX – TATA KELOLA PROYEK (DELIVERY GOVERNANCE)",
        "subs": ["9.1 Struktur Governance (Steering Committee, Sponsor, Manager, Tim)", "9.2 Mekanisme Pengambilan Keputusan", "9.3 Mekanisme Pelaporan Proyek"],
        "keywords": "governance steering committee sponsor decision making reporting",
        "length_intent": "Clear, operational, and structured. Use lists for roles. (Target: 250 words)."
    },
    {
        "id": "c_chap_10", "title": "BAB X – STRUKTUR DAN TIM PROYEK",
        "subs": ["10.1 Struktur Tim Konsultan", "10.2 Pengalaman dan Sertifikasi Tim"],
        "keywords": "project director manager sme technical analyst experience certification",
        "length_intent": "Professional profiles. Emphasize firm capability and historical experience. (Target: 300 words)."
    },
    {
        "id": "c_chap_11", "title": "BAB XI – MODEL PEMBIAYAAN (COMMERCIAL MODEL)",
        "subs": ["11.1 Total Biaya Proyek", "11.2 Model Pembayaran (Milestone)", "11.3 Durasi Kontrak", "11.4 Asumsi Proyek", "11.5 Batasan Pekerjaan (Scope Limitation)"],
        "keywords": "cost milestone contract duration assumption scope limitation commercial",
        "length_intent": "Highly precise. Use strict Markdown tables for cost breakdowns. (Target: 250 words)."
    },
    {
        "id": "c_chap_12", "title": "BAB XII – PENUTUP",
        "subs": ["12.1 Pernyataan Komitmen", "12.2 Harapan Kerja Sama", "12.3 Kontak Perusahaan"],
        "keywords": "commitment closing contact thank you",
        "length_intent": "Brief and impactful closing statement. (Target: 100 words)."
    }
]

PERSONAS = {
    "chap_1": "CEO & Board (Strategic, Urgent, Big Picture)",
    "c_chap_1": "CEO & Board (Strategic, Urgent, Big Picture)",
    "chap_3": "CTO/CIO (Technical, Methodical, Safe)",
    "c_chap_7": "CTO/CIO (Technical, Methodical, Safe)",
    "chap_8": "CFO/Procurement (Financial, ROI-focused, Strict)",
    "c_chap_11": "CFO/Procurement (Financial, ROI-focused, Strict)",
    "default": "Project Manager (Operational, Clear, Structured)"
}