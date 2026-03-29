"""Runtime and prompt configuration for the proposal generator."""
import os

# Backend mode and internal API settings.
DEMO_MODE = os.getenv("DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
_raw_data_mode = os.getenv("DATA_ACQUISITION_MODE", "demo" if DEMO_MODE else "staged").strip().lower()
DATA_ACQUISITION_MODE = "demo" if _raw_data_mode in {"demo", "legacy", "current"} else "staged"
FIRM_API_URL = "https://api.perusahaan-anda.com/v1"
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "isi_token_disini_nanti")

# External service configuration.
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "SERPER_API")
OLLAMA_HOST = "http://127.0.0.1:11434"

# Model and storage config.
LLM_MODEL = "gpt-oss:120b-cloud"
EMBED_MODEL = "bge-m3:latest"
DB_URI = "sqlite:///projects.db"

# Low-level shared testing runtime.
GENERATION_PROFILE = os.getenv("GENERATION_PROFILE", "balanced").strip().lower()
if GENERATION_PROFILE not in {"balanced", "throughput"}:
    GENERATION_PROFILE = "balanced"
# Each request still produces only one proposal document, but the server may
# run a small number of proposal jobs in parallel to protect end-to-end timing.
MAX_ACTIVE_GENERATIONS = max(1, int(os.getenv("MAX_ACTIVE_GENERATIONS", "2")))
MAX_GENERATION_BACKLOG = max(MAX_ACTIVE_GENERATIONS, int(os.getenv("MAX_GENERATION_BACKLOG", "18")))
JOB_RETENTION_SECONDS = max(300, int(os.getenv("JOB_RETENTION_SECONDS", "1800")))
JOB_POLL_INTERVAL_MS = max(1000, int(os.getenv("JOB_POLL_INTERVAL_MS", "2000")))
RESEARCH_CACHE_TTL_SECONDS = max(300, int(os.getenv("RESEARCH_CACHE_TTL_SECONDS", "1800")))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0").strip() or "0.0.0.0"
APP_PORT = max(1, int(os.getenv("APP_PORT", "5500")))

# Document length guardrails.
MAX_PROPOSAL_PAGES = 25
ESTIMATED_WORDS_PER_PAGE = 230
RESERVED_NON_CONTENT_PAGES = 2
PAGE_SAFETY_BUFFER = 1

# Writer identity.
WRITER_FIRM_NAME = "Inixindo Jogja"
DEFAULT_COLOR = (0, 51, 102)
WRITER_FIRM_OFFICE_ADDRESS = os.getenv("WRITER_FIRM_OFFICE_ADDRESS", "").strip()
WRITER_FIRM_EMAIL = os.getenv("WRITER_FIRM_EMAIL", "").strip()
WRITER_FIRM_PHONE = os.getenv("WRITER_FIRM_PHONE", "").strip()
WRITER_FIRM_WEBSITE = os.getenv("WRITER_FIRM_WEBSITE", "").strip()
WRITER_FIRM_CONTACT_INFO = os.getenv("WRITER_FIRM_CONTACT_INFO", "").strip()
WRITER_FIRM_PORTFOLIO = os.getenv("WRITER_FIRM_PORTFOLIO", "").strip()

COMPANY_DNA = {
    "positioning": (
        "Inixindo Jogja adalah mitra pembelajaran dan konsultasi yang membantu klien bergerak "
        "dari kebutuhan bisnis ke rencana eksekusi yang jelas, kredibel, dan dapat dijalankan."
    ),
    "proposal_promise": (
        "Proposal harus terasa seperti dokumen kerja yang siap dipakai untuk mengambil keputusan, "
        "bukan sekadar narasi yang menjelaskan situasi."
    ),
    "differentiators": [
        "Menerjemahkan kebutuhan bisnis ke pendekatan delivery yang rapi, terukur, dan mudah ditindaklanjuti.",
        "Menghubungkan metodologi, governance, dan komersial dalam satu alur keputusan yang konsisten.",
        "Menjaga bahasa proposal tetap profesional, singkat, dan relevan bagi sponsor eksekutif.",
    ],
    "client_value_focus": [
        "kejelasan keputusan",
        "kecepatan mobilisasi",
        "kontrol risiko dan tata kelola",
        "hasil bisnis yang terukur",
    ],
    "human_touch_review_points": [
        "kalibrasi relasi dan sponsor",
        "penajaman nilai komersial akhir",
        "pesan penutup yang paling personal dan relevan",
    ],
}

VALUE_PLAYBOOK = {
    "diagnostic": {
        "capability": "assessment, gap analysis, dan recommendation framing",
        "value_hook": "memberi kejelasan sebelum investasi diperbesar",
        "client_gains": [
            "baseline kondisi saat ini yang rapi",
            "prioritas masalah yang lebih tajam",
            "arah keputusan lanjutan yang lebih aman",
        ],
    },
    "strategic": {
        "capability": "strategy design, target operating model, dan roadmap prioritas",
        "value_hook": "menjembatani visi bisnis dengan urutan eksekusi yang realistis",
        "client_gains": [
            "arah transformasi yang lebih fokus",
            "prioritas investasi yang lebih defensible",
            "roadmap eksekusi yang lebih mudah disetujui",
        ],
    },
    "transformation": {
        "capability": "readiness assessment, rollout orchestration, change management, dan hypercare",
        "value_hook": "mengubah kebutuhan klien menjadi program perubahan yang lebih terkendali",
        "client_gains": [
            "mobilisasi yang lebih cepat",
            "kontrol risiko perubahan yang lebih baik",
            "adopsi hasil yang lebih stabil",
        ],
    },
    "implementation": {
        "capability": "solution delivery, implementation control, UAT readiness, dan handover",
        "value_hook": "membawa rancangan solusi menjadi hasil implementasi yang siap dipakai",
        "client_gains": [
            "go-live yang lebih tertata",
            "kualitas delivery yang lebih terjaga",
            "transisi ke operasi yang lebih rapi",
        ],
    },
}

INDUSTRY_VALUE_DRIVERS = {
    "Perbankan": [
        "pertumbuhan adopsi dan engagement nasabah",
        "ketahanan layanan digital",
        "kepatuhan terhadap regulasi dan kontrol risiko",
        "pengambilan keputusan yang lebih cepat dan terukur",
    ],
    "Telekomunikasi": [
        "service reliability",
        "pengendalian churn",
        "percepatan monetisasi layanan baru",
        "kontrol operasi lintas workstream",
    ],
    "Energi & Utilitas": [
        "reliability operasi",
        "kontrol risiko dan keselamatan",
        "prioritas eksekusi yang lebih jelas",
        "keberlanjutan hasil implementasi",
    ],
    "Pemerintah & BUMN": [
        "akuntabilitas keputusan",
        "tata kelola program",
        "keselarasan kebijakan dan eksekusi",
        "layanan yang lebih efektif",
    ],
}

DATA_MAPPING = {
    "entity": "Client Entity",
    "topic": "Strategic Initiative",
    "budget": "Investment Estimation"
}

PROJECT_DATA_FIELD_ALIASES = {
    "entity": [
        "entity", "client entity", "client_entity", "client", "client name", "client_name",
        "company", "company name", "company_name", "customer", "customer_name",
        "organization", "organization_name", "nama perusahaan", "nama_perusahaan",
        "nama klien", "nama_klien", "klien", DATA_MAPPING["entity"],
    ],
    "topic": [
        "topic", "strategic initiative", "strategic_initiative", "initiative", "initiative name",
        "project", "project name", "project_name", "program", "objective", "tujuan",
        "konteks organisasi", "konteks_organisasi", "inisiatif", DATA_MAPPING["topic"],
    ],
    "budget": [
        "budget", "investment estimation", "investment_estimation", "investment",
        "project budget", "project_budget", "estimated budget", "estimated_budget",
        "estimasi biaya", "estimasi_biaya", "anggaran", "nilai proyek", "nilai_proyek",
        DATA_MAPPING["budget"],
    ],
}

PROJECT_STANDARD_FIELD_ALIASES = {
    "methodology": [
        "methodology", "metodologi", "delivery methodology", "delivery_methodology",
        "approach", "framework", "service methodology", "metode kerja",
    ],
    "team": [
        "team", "team structure", "team_structure", "delivery team", "delivery_team",
        "resource plan", "resource_plan", "staffing", "struktur tim", "struktur_tim",
    ],
    "commercial": [
        "commercial", "komersial", "commercial terms", "commercial_terms",
        "pricing terms", "pricing_terms", "payment terms", "payment_terms",
        "scope and terms", "scope_terms",
    ],
}

FIRM_PROFILE_FIELD_ALIASES = {
    "office_address": [
        "office address", "office_address", "address", "alamat", "alamat kantor",
        "alamat_kantor", "office location", "office_location", "location",
    ],
    "email": [
        "email", "official email", "official_email", "company email", "company_email",
        "contact email", "contact_email",
    ],
    "phone": [
        "phone", "telephone", "telp", "telepon", "phone number", "phone_number",
        "contact number", "contact_number", "mobile",
    ],
    "website": [
        "website", "url", "company website", "company_website", "site", "web",
    ],
    "contact_info": [
        "contact info", "contact_info", "contact", "kontak", "informasi kontak",
        "informasi_kontak",
    ],
    "portfolio_highlights": [
        "portfolio highlights", "portfolio_highlights", "portfolio", "highlight portfolio",
        "highlight_portfolio", "capabilities", "company profile", "company_profile",
    ],
}

CLIENT_RELATIONSHIP_FIELD_ALIASES = {
    "summary": [
        "summary", "relationship summary", "relationship_summary", "description",
        "notes", "detail", "history", "engagement history", "engagement_history",
    ],
    "status": [
        "status", "relationship status", "relationship_status", "mode", "client_status",
        "engagement_status", "has_relationship",
    ],
}

# Keyword/regulation suggestions used by the UI.
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

# Fallback firm data used in demo mode.
MOCK_FIRM_PROFILE = {
    "office_address": WRITER_FIRM_OFFICE_ADDRESS,
    "email": WRITER_FIRM_EMAIL,
    "phone": WRITER_FIRM_PHONE,
    "website": WRITER_FIRM_WEBSITE,
    "contact_info": WRITER_FIRM_CONTACT_INFO,
    "portfolio_highlights": WRITER_FIRM_PORTFOLIO
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

# Standard chapter structure (max 25 pages total).
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
    },
    {
        "id": "c_closing", "title": "PENUTUP & APRESIASI KEMITRAAN",
        "subs": ["Apresiasi dan Komitmen Kemitraan", "Informasi Kontak dan Langkah Lanjutan"],
        "keywords": "closing appreciation partnership contact next steps",
        "length_intent": "Tutup proposal dengan apresiasi profesional, pernyataan komitmen kemitraan, informasi kontak lengkap, dan ajakan langkah tindak lanjut yang jelas. (Target: 350 words)."
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
    "c_closing": "Client Engagement Partner",
    "default": "Principal Management Consultant"
}

PROPOSAL_SYSTEM_PROMPT = """
You are a Principal Consultant and Technical Writer at {writer_firm}.
Your target audience is the Executive Board of {client}. Adopt this persona: {persona}.

Writing rules:
1. Be concise and direct ("singkat, padat, tidak bertele-tele"), but keep the content specific and useful.
2. Formatting:
   - Use EXACT heading structure from provided H2 list.
   - Under each H2, create 2-3 H3 subsections (###) that are relevant and non-redundant.
   - Use numbered lists for steps/sequences (1., 2., 3.) and bullet lists for supporting detail points (-).
   - Keep prose dense: avoid excessive blank lines and avoid one-line bullets without explanation.
   - Include at least 1 markdown table only for chapters with operational, governance, timeline, or commercial content.
3. Depth: Explain rationale, assumptions, risks, dependencies, metrics, and expected deliverables in actionable detail.
4. Tone: Professional, objective, direct, and persuasive in Indonesian. Avoid academic filler.
5. Synthesis: Integrate OSINT data naturally as facts, and do not mention "berdasarkan sumber online".
6. Citations (APA in-text):
   - For external claims, use the provided source hints and cite like (domain.tld, Year).
   - Never use placeholder citations such as (OSINT #1), (OSINT_PROFILE #2), or (RAG Semantic).
   - For internal data claims (historical/semantic), use (Data Internal, {current_year}).

Context data:
Global OSINT Data (includes URL + APA hint): {global_data}
Client News OSINT (includes URL + APA hint): {client_news}
Regulatory OSINT (includes URL + APA hint): {regulation_data}
Historical Internal Data: {structured_row_data}
Semantic Internal Data: {rag_data}

Additional instructions:
{extra_instructions}

Task:
Write "{chapter_title}" ensuring it meets the length intent: {length_intent}
Include these H2 (##) exactly:
{sub_chapters}

Before finalizing, check:
- Provide concrete examples relevant to the client context.
- Ensure all key claims have business/technical rationale.
- Ensure the chapter is sufficiently detailed for board-level review and contributes proportionally to a total proposal maximum 25 pages (avoid over-expansion).

{visual_prompt}
Do not write greetings or introductions. Output only the chapter content.
"""
