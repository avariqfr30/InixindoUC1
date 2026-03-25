"""Flask entrypoint for the proposal generator app."""
import io
import logging
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from config import DB_URI, SMART_SUGGESTIONS
from core import FinancialAnalyzer, KnowledgeBase, ProposalGenerator

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = Flask(__name__)
CORS(app)

knowledge_base = KnowledgeBase(DB_URI)
proposal_generator = ProposalGenerator(knowledge_base)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/config')
def get_base_config():
    return jsonify({"suggestions": SMART_SUGGESTIONS})

@app.route('/api/companies')
def get_companies():
    if knowledge_base.df is None or knowledge_base.df.empty:
        return jsonify([])
    companies = knowledge_base.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    companies = [c for c in companies if c.lower() != 'nan' and c]
    return jsonify(sorted(companies))

@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget():
    """Estimate pricing tiers from public financial signals."""
    data = request.json or {}
    client_name = data.get('nama_perusahaan', '')

    analyzer = FinancialAnalyzer(proposal_generator.ollama)
    result = analyzer.suggest_budget(
        client_name=client_name,
        timeline=data.get('estimasi_waktu', ''),
        project_type=data.get('jenis_proyek', ''),
        service_type=data.get('jenis_proposal', ''),
        project_goal=data.get('klasifikasi_kebutuhan', ''),
        objective=data.get('konteks_organisasi', ''),
        notes=data.get('permasalahan', ''),
        frameworks=data.get('potensi_framework', ''),
    )
    return jsonify(result)

@app.route('/api/preview-outline', methods=['POST'])
def preview_outline():
    data = request.json or {}
    outline = proposal_generator.build_preview_outline(data)
    return jsonify({"outline": outline})

@app.route('/generate', methods=['POST'])
def generate_proposal():
    data = request.json or {}
    required_fields = [
        'nama_perusahaan',
        'konteks_organisasi',
        'estimasi_biaya',
        'jenis_proposal',
        'klasifikasi_kebutuhan',
        'jenis_proyek',
        'estimasi_waktu',
        'permasalahan',
        'potensi_framework',
    ]
    missing = [f for f in required_fields if not str(data.get(f, '')).strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        doc, filename = proposal_generator.generate_document(
            client=data['nama_perusahaan'],
            project=data['konteks_organisasi'],
            budget=data['estimasi_biaya'],
            service_type=data['jenis_proposal'],
            project_goal=data['klasifikasi_kebutuhan'],
            project_type=data['jenis_proyek'],
            timeline=data['estimasi_waktu'],
            notes=data['permasalahan'],
            regulations=data['potensi_framework'],
            chapter_id=data.get('chapter_id')
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    out = io.BytesIO()
    doc.save(out)
    out.seek(0)

    return send_file(
        out,
        as_attachment=True,
        download_name=f"{filename}.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

@app.route('/refresh-knowledge', methods=['POST'])
def refresh_knowledge():
    success = knowledge_base.refresh_data()
    return jsonify({"status": "success" if success else "error"})

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)
