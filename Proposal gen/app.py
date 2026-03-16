"""
Web Server Gateway Interface (WSGI) application.
Exposes REST endpoints for the frontend interface to trigger the proposal generation engine.
"""

import io
import logging
from typing import Any
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.wrappers import Response

from core import ProposalGenerator, KnowledgeBase
from config import DB_URI

# Configure application-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize singletons
knowledge_base = KnowledgeBase(DB_URI)
generator = ProposalGenerator(knowledge_base)

@app.route('/')
def home() -> str:
    """Serves the primary UI."""
    return render_template('index.html')

@app.route('/api/companies', methods=['GET'])
def get_companies() -> Response:
    """Returns a unique, sorted list of known companies from the vector DB."""
    if knowledge_base.dataframe is None or knowledge_base.dataframe.empty: 
        return jsonify([])
        
    companies = knowledge_base.dataframe['entity'].dropna().astype(str).str.strip().unique().tolist()
    clean_companies = sorted([comp for comp in companies if comp.lower() != 'nan' and comp])
    
    return jsonify(clean_companies)

@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget_endpoint() -> Response:
    """Invokes the async OSINT financial search to formulate a budget suggestion."""
    data = request.json
    client_name = data.get('nama_perusahaan')
    
    if not client_name:
        return jsonify({"suggestion": "Silakan pilih entitas perusahaan terlebih dahulu."})
        
    logger.info(f"Computing financial budget suggestion for: {client_name}")
    suggestion = generator.suggest_budget(client_name)
    return jsonify({"suggestion": suggestion})

@app.route('/generate', methods=['POST'])
def generate_document() -> Any:
    """
    Primary ingestion endpoint.
    Expects a JSON payload from the classic UI template.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        client_name = data.get('nama_perusahaan', 'Unknown Client')
        logger.info(f"Received generation request for client: {client_name}")
        
        doc, filename = generator.run(
            client=client_name,
            project_status=data.get('status_proyek', 'Proyek Baru'),
            project_type=data.get('jenis_proyek', 'Implementation'),
            scope=data.get('ruang_lingkup', ''),
            project_goal=data.get('outcome', 'Peningkatan Sistem IT'),
            timeline=data.get('estimasi_waktu', 'TBD'),
            budget=data.get('target_budget', 'TBD'),
            regulations=data.get('regulasi', ''),
            notes=data.get('catatan_tambahan', '')
        )
        
        output_stream = io.BytesIO()
        doc.save(output_stream)
        output_stream.seek(0)
        
        return send_file(
            output_stream, 
            as_attachment=True, 
            download_name=f"{filename}.docx", 
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    except Exception as error:
        logger.exception("Catastrophic failure during document generation.")
        return jsonify({"error": "Terjadi kesalahan internal pada server AI.", "details": str(error)}), 500

@app.route('/refresh-knowledge', methods=['POST'])
def refresh_vectors() -> Response:
    """Forces the KnowledgeBase to re-sync with local SQL/CSV files."""
    success = knowledge_base.refresh_data()
    return jsonify({"status": "success" if success else "error"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, threaded=True)