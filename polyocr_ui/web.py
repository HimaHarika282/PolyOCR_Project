from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from ocr_processing import process_printed_image_with_easyocr
from handwritten_ocr import process_paragraph_image 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'
JSON_FOLDER = 'static/json_output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/detect_text', methods=['POST'])
def detect_text():
    print("Route reached")

    ocr_type = request.form.get('ocr_type')
    image = request.files.get('image')

    print("OCR Type received:", ocr_type)
    print("Image received:", "Yes" if image else "No")

    if not ocr_type or not image:
        return jsonify({'error': 'Missing OCR type or image'}), 400


    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
    image.save(image_path)

    try:
        if ocr_type == 'Printed Text':
            result = process_printed_image_with_easyocr(image_path)
        elif ocr_type == 'Handwritten Text':
             try:
                    result = process_paragraph_image(image_path)
             except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Handwritten OCR failed: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid OCR type selected.'}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
