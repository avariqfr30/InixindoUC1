# app.py
import io
import logging
from flask import Flask, send_file, request, jsonify, render_template
from flask_cors import CORS

from core import ProposalGenerator, KnowledgeBase
from config import DB_FILE

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
    main_col = kb.df.columns[0]
    sub_col = kb.df.columns[1]
    
    for entity in kb.df[main_col].unique():
        options = kb.df[kb.df[main_col] == entity][sub_col].unique().tolist()
        tree[entity] = {"label": entity, "options": options}
        
    return jsonify({"structure": tree, "labels": [main_col, sub_col]})

@app.route('/generate', methods=['POST'])
def generate_doc():
    data = request.json
    doc, filename = generator.run(data['topic'], data['sub_topic'])
    
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