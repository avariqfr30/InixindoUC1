# app.py
import io
import logging
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS
from core import ProposalGenerator, KnowledgeBase

# FIXED: Removed SMART_SUGGESTIONS from the import list
from config import DB_URI, DATA_MAPPING 

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

kb = KnowledgeBase(DB_URI)
generator = ProposalGenerator(kb)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/companies')
def get_companies():
    if kb.df is None or kb.df.empty: return jsonify([])
    companies = kb.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    return jsonify(sorted([c for c in companies if c.lower() != 'nan' and c]))

@app.route('/generate', methods=['POST'])
def generate_doc():
    data = request.json
    
    doc, filename = generator.run(
        client=data.get('nama_perusahaan'),
        industry=data.get('sektor_industri'),
        employee_count=data.get('jumlah_karyawan'),
        dm_age=data.get('usia_pimpinan'),
        project_status=data.get('status_proyek'),
        project_type=data.get('jenis_proyek'),
        scope=data.get('ruang_lingkup'),
        outcome=data.get('outcome'),
        regulations=data.get('regulasi'),
        timeline=data.get('estimasi_waktu'),
        budget=data.get('target_budget'),
        notes=data.get('catatan_tambahan')
    )
    
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