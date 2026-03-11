# app.py
import io
import logging
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS

from core import ProposalGenerator, KnowledgeBase
from config import DB_URI, DATA_MAPPING, SMART_SUGGESTIONS

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
    """Returns static configurations and intelligent suggestions."""
    return jsonify({
        "suggestions": SMART_SUGGESTIONS
    })

@app.route('/api/companies')
def get_companies():
    """Scalable: Only extracts and returns unique company names."""
    if kb.df is None or kb.df.empty: 
        return jsonify([])
    
    # Fast pandas extraction for 30k+ rows
    companies = kb.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    companies = [c for c in companies if c.lower() != 'nan' and c]
    return jsonify(sorted(companies))

@app.route('/api/projects')
def get_projects():
    """Scalable: Only returns projects for the specifically requested company."""
    entity = request.args.get('entity')
    if not entity or kb.df is None or kb.df.empty: 
        return jsonify({})
    
    # Vectorized filtering for high performance
    subset = kb.df[kb.df['entity'].astype(str).str.strip() == entity]
    
    projects = {}
    # to_dict('records') is much faster than iterrows()
    for row in subset.to_dict('records'):
        topic = str(row.get('topic', '')).strip()
        if topic and topic.lower() != 'nan':
            projects[topic] = {
                "permasalahan": str(row.get('Strategic Context & Pain Points', '')).strip(),
                "biaya": str(row.get('budget', '')).strip()
            }
            
    return jsonify(projects)

@app.route('/generate', methods=['POST'])
def generate_doc():
    data = request.json
    
    # Map the new frontend payload back to the core generator variables
    entity = data.get('nama_perusahaan')
    topic = data.get('konteks_organisasi')
    budget = data.get('estimasi_biaya')
    service_type = data.get('jenis_proposal')
    
    project_goal = "Improvement" # Default assumption
    project_type = data.get('klasifikasi_kebutuhan', 'Implementation')
    timeline = data.get('estimasi_waktu', 'TBD')
    notes = data.get('permasalahan', '')
    regulations = data.get('potensi_framework', '')
    
    doc, filename = generator.run(entity, topic, budget, service_type, project_goal, project_type, timeline, notes, regulations)
    
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