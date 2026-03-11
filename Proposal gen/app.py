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

@app.route('/get-config')
def get_config():
    if kb.df is None: 
        return jsonify({"error": "DB Load Failed"}), 500
        
    tree = {}
    # Dynamically build the options based on the dataset
    for _, row in kb.df.iterrows():
        entity = str(row.get('entity', '')).strip()
        if not entity or entity == 'nan': continue
        
        if entity not in tree:
            tree[entity] = {
                "konteks": set(),
                "permasalahan": set(),
                "biaya": set()
            }
            
        # Mapping DB columns to the new UI categories
        topic = str(row.get('topic', '')).strip()
        problem = str(row.get('Strategic Context & Pain Points', '')).strip() # Using db.csv column
        budget = str(row.get('budget', '')).strip()
        
        if topic and topic != 'nan': tree[entity]["konteks"].add(topic)
        if problem and problem != 'nan': tree[entity]["permasalahan"].add(problem)
        if budget and budget != 'nan': tree[entity]["biaya"].add(budget)
            
    # Convert sets to lists for JSON serialization
    for e in tree:
        tree[e]["konteks"] = list(tree[e]["konteks"])
        tree[e]["permasalahan"] = list(tree[e]["permasalahan"])
        tree[e]["biaya"] = list(tree[e]["biaya"])
            
    return jsonify({"structure": tree, "suggestions": SMART_SUGGESTIONS})

@app.route('/generate', methods=['POST'])
def generate_doc():
    data = request.json
    
    # Map the new frontend payload back to the core generator variables
    entity = data.get('nama_perusahaan')
    topic = data.get('konteks_organisasi')
    budget = data.get('estimasi_biaya')
    service_type = data.get('jenis_proposal')
    
    project_goal = "Improvement" # Default assumption or can be added back to UI
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