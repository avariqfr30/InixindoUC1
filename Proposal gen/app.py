"""
Main entry point for the Flask web application.
"""
import io
import logging
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS

from core import ProposalGenerator, KnowledgeBase, FinancialAnalyzer
from config import DB_URI, SMART_SUGGESTIONS

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

kb = KnowledgeBase(DB_URI)
generator = ProposalGenerator(kb)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/config')
def get_base_config():
    return jsonify({"suggestions": SMART_SUGGESTIONS})

@app.route('/api/companies')
def get_companies():
    if kb.df is None or kb.df.empty: 
        return jsonify([])
    companies = kb.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    companies = [c for c in companies if c.lower() != 'nan' and c]
    return jsonify(sorted(companies))

@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget():
    """Endpoint for OSINT based Smart Financial Pricing"""
    data = request.json
    client_name = data.get('nama_perusahaan', '')
    
    analyzer = FinancialAnalyzer(generator.ollama)
    result = analyzer.suggest_budget(client_name)
    return jsonify(result)

@app.route('/api/preview-outline', methods=['POST'])
def preview_outline():
    data = request.json or {}
    outline = generator.build_preview_outline(data)
    return jsonify({"outline": outline})

@app.route('/generate', methods=['POST'])
def generate_doc():
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
        doc, filename = generator.run(
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
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
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
def refresh():
    success = kb.refresh_data()
    return jsonify({"status": "success" if success else "error"})

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)
