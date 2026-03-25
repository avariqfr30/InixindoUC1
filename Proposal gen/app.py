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
    data = request.json
    try:
        doc, filename = generator.run(
            client=data.get('nama_perusahaan', ''),
            project=data.get('konteks_organisasi', ''),
            budget=data.get('estimasi_biaya', ''),
            service_type=data.get('jenis_proposal', 'Konsultan'),
            project_goal=data.get('klasifikasi_kebutuhan', 'Problem'),
            project_type=data.get('jenis_proyek', 'Implementation'),
            timeline=data.get('estimasi_waktu', 'TBD'),
            notes=data.get('permasalahan', ''),
            regulations=data.get('potensi_framework', ''),
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
