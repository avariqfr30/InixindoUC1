# config.py
import os

# =====================================================================
# SYSTEM MODE & API ADAPTER
# =====================================================================
DEMO_MODE = True 
FIRM_API_URL = "https://api.perusahaan-anda.com/v1" 
API_AUTH_TOKEN = "isi_token_disini_nanti"

GOOGLE_API_KEY = "API_KEY"
GOOGLE_CX_ID = "CX_ID"
OLLAMA_HOST = "http://127.0.0.1:11434"
LLM_MODEL = "gpt-oss:120b-cloud" 
EMBED_MODEL = "bge-m3:latest"
DB_URI = "sqlite:///projects.db" 

WRITER_FIRM_NAME = "Inixindo Jogja" 
DEFAULT_COLOR = (0, 51, 102)

DATA_MAPPING = { "entity": "Client Entity", "topic": "Strategic Initiative", "budget": "Investment Estimation" }

MOCK_FIRM_PROFILE = {
    "contact_info": "Kantor Pusat Inixindo Jogja\nJl. Kenari No. 69, Muja Muju, Kec. Umbulharjo, Kota Yogyakarta, DIY 55165\nEmail: info@inixindo.id\nTelp: (0274) 515448",
    "portfolio_highlights": "Inixindo Jogja memiliki pengalaman lebih dari 30 tahun dalam Transformasi TI, Audit, dan Pelatihan Enterprise."
}

MOCK_FIRM_STANDARDS = {
    "Diagnostic": {"methodology": "1. Discovery\n2. As-Is Analysis\n3. Gap Analysis\n4. Report", "team": "1x Principal Consultant, 1x Business Analyst.", "commercial": "Fixed-fee based on scope."},
    "Strategic": {"methodology": "1. Executive Alignment\n2. To-Be Visioning\n3. Roadmap", "team": "1x Project Director, 1x Enterprise Architect.", "commercial": "Retainer or Fixed-fee."},
    "Transformation": {"methodology": "1. Readiness Assessment\n2. Phased Rollout\n3. OCM\n4. Hypercare", "team": "1x Program Manager, 2x Domain Leads.", "commercial": "Time & Materials (T&M)."},
    "Implementation": {"methodology": "1. Design\n2. UAT & Testing\n3. Deployment\n4. Go-Live", "team": "1x Project Manager, 1x Lead Engineer.", "commercial": "Milestone-based (20% Kickoff, 40% UAT, 30% Go-Live, 10% Handover)."}
}

# =====================================================================
# TONE OF VOICE MAPPING (Psychographic Targeting based on DM Age)
# =====================================================================
TONE_MAPPINGS = {
    "Boomer (>60 Tahun)": "Sangat formal, penuh rasa hormat, berorientasi pada mitigasi risiko, keamanan, dan stabilitas jangka panjang. Hindari jargon teknis yang tidak perlu.",
    "Gen X (45 - 60 Tahun)": "Profesional, terstruktur, berorientasi pada ROI yang jelas, efisiensi proses, dan kepatuhan (compliance).",
    "Millennial (30 - 45 Tahun)": "Modern, langsung pada intinya (to the point), berorientasi pada skalabilitas, inovasi teknologi, dan kecepatan eksekusi.",
    "Gen Z (<30 Tahun)": "Sangat tajam, ringkas, visioner, berorientasi pada disrupsi digital, UI/UX, dan fleksibilitas (Agile)."
}

# =====================================================================
# PROPOSAL STRUCTURE (Anti-Bertele-tele & Decoy Pricing)
# =====================================================================
UNIVERSAL_STRUCTURE = [
    {
        "id": "c_1", "title": "BAB I – KONTEKS ORGANISASI",
        "subs": ["1.1 Latar Belakang Perusahaan", "1.2 Dinamika dan Kebutuhan Industri"],
        "keywords": "context background vision strategy",
        "length_intent": "Sharp and concise. Maximum 2 short paragraphs per section. Focus on the provided Client Scope and Industry."
    },
    {
        "id": "c_2", "title": "BAB II – RUANG LINGKUP & PERMASALAHAN",
        "subs": ["2.1 Tantangan Utama", "2.2 Ruang Lingkup Pekerjaan (Scope)"],
        "keywords": "problem pain points bottleneck issue scope",
        "length_intent": "Use BULLET POINTS exclusively for the Scope. Be extremely clear about what is included. Do not write long paragraphs."
    },
    {
        "id": "c_3", "title": "BAB III – OUTCOME DAN SOLUSI",
        "subs": ["3.1 Target Pencapaian (Goals)", "3.2 Arsitektur Pendekatan Solusi"],
        "keywords": "solution approach goal outcome objective",
        "length_intent": "Highlight the Tangible Impact (Outcome) using bold metrics. Connect the solution directly to the goals provided."
    },
    {
        "id": "c_4", "title": "BAB IV – KEPATUHAN DAN REGULASI",
        "subs": ["4.1 Framework Utama", "4.2 Pemenuhan Regulasi"],
        "keywords": "framework iso cobit itil regulation compliance",
        "length_intent": "List the frameworks using bullet points. Briefly state why they matter for this specific industry."
    },
    {
        "id": "c_6", "title": "BAB V – METODOLOGI & JADWAL",
        "subs": ["5.1 Pendekatan Pelaksanaan", "5.2 Estimasi Waktu (Timeline)"],
        "keywords": "methodology timeline phase schedule",
        "length_intent": "Use numbered lists for the methodology phases. Map them strictly to the estimated timeline provided."
    },
    {
        "id": "c_9", "title": "BAB VI – ESTIMASI INVESTASI (PRICING)",
        "subs": ["6.1 Opsi Investasi Tersedia", "6.2 Ketentuan Komersial"],
        "keywords": "budget cost commercial investment terms decoy pricing",
        "length_intent": """CRITICAL INSTRUCTION - DECOY PRICING:
Create a Markdown table with EXACTLY 3 Tiers of pricing based on the provided Target Budget:
1. Opsi Esensial: 20% lower than target budget. Minimum scope, lacks premium support.
2. Opsi Rekomendasi: Matches Target Budget exactly. Highest perceived value, includes standard support.
3. Opsi Premium: 60% higher than target budget. Anchoring option with 24/7 VIP Hypercare, extensive warranties.
Make the 'Rekomendasi' tier look the most logical."""
    },
    {
        "id": "c_10", "title": "BAB VII – PENUTUP",
        "subs": ["7.1 Kesimpulan", "7.2 Kontak Resmi"],
        "keywords": "closing commitment next steps contact",
        "length_intent": "One short paragraph conclusion. End with the official firm contact info."
    }
]

PROPOSAL_SYSTEM_PROMPT = """
You are an elite Proposal Architect for {writer_firm}.
Client: {client} | Industry: {industry} | Employees: {employee_count}
Project Type: {project_status} - {project_type}

--- PSYCHOGRAPHIC TONE ---
The Decision Maker is in this age group: {dm_age}.
Your writing style MUST strictly follow this tone: {tone_instruction}

--- ANTI-BERTELE-TELE (CONCISENESS) RULE ---
1. DO NOT BE LONG-WINDED. Executives hate fluff.
2. Be "Tajam" (Sharp) and to the point.
3. Use bullet points heavily for lists, scope, and impact.
4. Use bold text to highlight key numbers or outcomes.
5. Limit paragraphs to 2-3 sentences max.

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

--- TASK ---
Write "{chapter_title}".
Include these sub-chapters exactly as H2 (##):
{sub_chapters}
Follow the length intent: {length_intent}
"""