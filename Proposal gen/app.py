# app.py
import io
import logging
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS

from core import ProposalGenerator, KnowledgeBase
from config import DB_FILE, DATA_MAPPING

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

kb = KnowledgeBase(DB_FILE)
generator = ProposalGenerator(kb)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/get-config')
def get_config():
    if kb.df is None: 
        return jsonify({"error": "DB Load Failed"}), 500
        
    tree = {}
    
    for entity in kb.df['entity'].dropna().unique():
        tree[entity] = {"label": entity, "topics": {}}
        
        entity_df = kb.df[kb.df['entity'] == entity]
        
        for topic in entity_df['topic'].dropna().unique():
            topic_df = entity_df[entity_df['topic'] == topic]
            budgets = topic_df['budget'].dropna().unique().tolist()
            
            tree[entity]["topics"][topic] = {
                "label": topic,
                "budgets": budgets
            }
            
    return jsonify({
        "structure": tree, 
        "labels": [DATA_MAPPING["entity"], DATA_MAPPING["topic"], DATA_MAPPING["budget"]] 
    })

@app.route('/generate', methods=['POST'])
def generate_doc():
    data = request.json
    
    entity = data.get('entity')
    topic = data.get('topic')
    budget = data.get('budget')
    service_type = data.get('service_type') # Capture the new UI selection
    
    doc, filename = generator.run(entity, topic, budget, service_type)
    
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