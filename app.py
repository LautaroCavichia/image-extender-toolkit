# app.py

from flask import Flask, render_template, request, send_file
import os
from PIL import Image
import uuid

from remove_bg import remove_background
from create_pattern import create_pattern
from extend import extend
from extend_ai import extend_ai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    unique_id = str(uuid.uuid4())
    extension = file.filename.rsplit('.', 1)[-1].lower()
    filename = f"{unique_id}.{extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return {'file_id': unique_id, 'extension': extension}, 200

@app.route('/process/<action>', methods=['POST'])
def process_image(action):
    file_id = request.form.get('file_id')
    extension = request.form.get('extension')
    prompt = request.form.get('prompt')  # Retrieve the prompt from the request
    if not file_id or not extension:
        return "Invalid request", 400
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{extension}")
    
    try:
        image = Image.open(input_path)
    except Exception as e:
        return f"Error opening image: {str(e)}", 500
    
    # Routing the action to the correct function
    if action == 'remove_bg':
        processed_image = remove_background(image)
    elif action == 'uncrop':
        processed_image = extend(image)
    elif action == 'create_pattern':
        processed_image = create_pattern(image)
    elif action == 'extend_ai':
        if not prompt:
            return "Prompt is required for extend_ai", 400
        processed_image = extend_ai(image, prompt)
    else:
        return "Invalid action", 400
    
    # Save processed result
    output_filename = f"{file_id}_{action}.{extension}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    processed_image.save(output_path)
    
    # Return it as an attachment (download)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)