    # Lexi - An OCR for Kannada Handwritten Text
    # Copyright (C) 2025  Sahil Kumar Jamwal

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

    # Contact: Sahil Kumar Jamwal
    # Email: sahilkumarjamwal464@gmail.com
    # GitHub: https://github.com/precious-soda

import os
import uuid
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
import importlib
import tempfile
from pdf2image import convert_from_path
import shutil
import weasyprint
from weasyprint import HTML, CSS
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

UPLOAD_FOLDER = './uploads'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

KANNADA_FONT_PATH = './models/NotoSansKannada-VariableFont_wdthwght.ttf'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_line_and_word_images(segmentation_result, static_folder, session_id, page_index=None):
    session_folder = os.path.join(static_folder, session_id)
    lines_folder = os.path.join(session_folder, 'lines')
    words_folder = os.path.join(session_folder, 'words')
    
    os.makedirs(session_folder, exist_ok=True)
    os.makedirs(lines_folder, exist_ok=True)
    os.makedirs(words_folder, exist_ok=True)
    
    for line_id in segmentation_result["lines"]:
        line_data = segmentation_result["lines"][line_id]
        prefix = f"page_{page_index}_" if page_index is not None else ""
        
        if line_data.get("line_img") is not None:
            try:
                line_filename = f"{prefix}line_{line_id}.png"
                line_path = os.path.join(lines_folder, line_filename)
                cv2.imwrite(line_path, line_data["line_img"])
                line_data["line_url"] = f"/static/{session_id}/lines/{line_filename}"
                print(f"Saved line image: {line_path}")
            except Exception as e:
                print(f"Error saving line {line_id} image: {e}")
                line_data["line_url"] = None
        else:
            line_data["line_url"] = None
        
        if "words" in line_data:
            for word_id in line_data["words"]:
                word_data = line_data["words"][word_id]
                
                if word_data.get("word_img") is not None:
                    try:
                        word_filename = f"{prefix}line_{line_id}_word_{word_id}.png"
                        word_path = os.path.join(words_folder, word_filename)
                        cv2.imwrite(word_path, word_data["word_img"])
                        word_data["word_url"] = f"/static/{session_id}/words/{word_filename}"
                        print(f"Saved word image: {word_path}")
                    except Exception as e:
                        print(f"Error saving line {line_id} word {word_id} image: {e}")
                        word_data["word_url"] = None
                else:
                    word_data["word_url"] = None

def save_predicted_text(text, static_folder, session_id, page_index=0):
    session_folder = os.path.join(static_folder, session_id)
    os.makedirs(session_folder, exist_ok=True)
    text_filename = f"predicted_text_page_{page_index}.txt"
    text_path = os.path.join(session_folder, text_filename)
    if text.startswith('/static/'):
        print(f"Error: Attempted to save URL as text: {text}")
        raise ValueError("Predicted text is a URL, not transcribed text")
    print(f"Writing to {text_path} with content:\n{text}")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved predicted text: {text_path}")
    return f"/static/{session_id}/{text_filename}"

def generate_html_content(text_files, session_id, metrics=None):
    """Generate HTML content for the PDF with proper Kannada text styling and formatted metrics"""
    
    css_content = f"""
    @page {{
        margin: 2cm;
        size: A4;
        @bottom-center {{
            content: "Page " counter(page);
            font-size: 10px;
            color: #888;
        }}
    }}
    
    body {{
        font-family: 'Noto Sans Kannada', Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #333;
        margin: 0;
        padding: 0;
    }}
    
    .page {{
        page-break-after: always;
        min-height: calc(297mm - 4cm); /* A4 height minus margins */
    }}
    
    .page:last-child {{
        page-break-after: avoid;
    }}
    
    .page-header {{
        font-size: 12px;
        color: #666;
        margin-bottom: 1cm;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5cm;
    }}
    
    .text-content {{
        white-space: pre-wrap;
        word-wrap: break-word;
        text-align: justify;
    }}
    
    .metrics-section {{
        margin-top: 0cm;
        padding: 1cm;
        display: flex;
        flex-direction: row;
        gap: 1cm;
    }}
    
    .metric-box {{
        flex: 1;
        padding: 1cm;
        border-radius: 4px;
    }}
    
    .metric-box.cer {{
        background-color: #EBF5FB; /* Matches bg-blue-50 */
    }}
    
    .metric-box.wer {{
        background-color: #F0FFF4; /* Matches bg-green-50 */
    }}
    
    .metric-box h2 {{
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 0.5cm;
    }}
    
    .metric-box .percentage {{
        font-size: 24px;
        font-weight: bold;
        margin: 0.5cm 0;
    }}
    
    .metric-box .cer .percentage {{
        color: #2563EB; /* Matches text-blue-600 */
    }}
    
    .metric-box .wer .percentage {{
        color: #059669; /* Matches text-green-600 */
    }}
    
    .metric-box p {{
        font-size: 12px;
        color: #4B5563; /* Matches text-gray-600 */
        margin: 0;
    }}
    
    .metric-box .cer h2 {{
        color: #1E40AF; /* Matches text-blue-800 */
    }}
    
    .metric-box .wer h2 {{
        color: #065F46; /* Matches text-green-800 */
    }}
    """
    
    if os.path.exists(KANNADA_FONT_PATH):
        css_content = f"""
        @font-face {{
            font-family: 'Noto Sans Kannada';
            src: url('file://{os.path.abspath(KANNADA_FONT_PATH)}') format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """ + css_content
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="kn">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transcribed Text - {session_id}</title>
        <style>
            {css_content}
        </style>
    </head>
    <body>
    """
    
    for i, text_path in enumerate(text_files):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading text file {text_path}: {e}")
            text = f"Error: Could not read text from {os.path.basename(text_path)}"
        
        import html
        escaped_text = html.escape(text)
        
        html_content += f"""
        <div class="page">
            <div class="text-content">{escaped_text}</div>
        </div>
        """
    
    # Append metrics section if metrics are provided
    if metrics:
        html_content += f"""
        <div class="page">
            <div class="metrics-section">
                <div class="metric-box cer">
                    <h2>Character Error Rate</h2>
                    <div class="percentage">{metrics.get('CER', {}).get('percentage', 'N/A')}%</div>
                    <p>Edit distance: {metrics.get('CER', {}).get('editDistance', 'N/A')}/{metrics.get('CER', {}).get('totalCharacters', 'N/A')}</p>
                </div>
                <div class="metric-box wer">
                    <h2>Word Error Rate</h2>
                    <div class="percentage">{metrics.get('WER', {}).get('percentage', 'N/A')}%</div>
                    <p>Edit distance: {metrics.get('WER', {}).get('editDistance', 'N/A')}/{metrics.get('WER', {}).get('totalWords', 'N/A')}</p>
                </div>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        if 'model' not in request.form:
            return jsonify({"error": "No model selected"}), 400
        
        file = request.files['file']
        model_type = request.form['model']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            session_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            
            session_static_folder = os.path.join(app.config['STATIC_FOLDER'], session_id)
            os.makedirs(session_static_folder, exist_ok=True)
            os.makedirs(os.path.join(session_static_folder, 'lines'), exist_ok=True)
            os.makedirs(os.path.join(session_static_folder, 'words'), exist_ok=True)
            os.makedirs(os.path.join(session_static_folder, 'chars'), exist_ok=True)
            
            try:
                model_module = importlib.import_module({
                    'densenet121_attention': 'densenet',
                    'botnet_resnet50': 'botnet',
                    'vit_base': 'vit'
                }.get(model_type, 'invalid'))
                if model_module == 'invalid':
                    os.remove(file_path)
                    return jsonify({"error": f"Unsupported model type: {model_type}"}), 400
            except ImportError as e:
                os.remove(file_path)
                return jsonify({"error": f"Failed to import model module: {str(e)}"}), 500
            
            try:
                main_model = model_module.main_model
                ott_model = model_module.ott_model
                main_mean = model_module.main_mean
                main_std = model_module.main_std
                ott_mean = model_module.ott_mean
                ott_std = model_module.ott_std
                main_class_names = model_module.main_class_names
                ott_class_names = model_module.ott_class_names
            except AttributeError as e:
                os.remove(file_path)
                return jsonify({"error": f"Model module missing required attributes: {str(e)}"}), 500
            
            response_data = {}
            temp_files = []
            
            if filename.lower().endswith('.pdf'):
                try:
                    images = convert_from_path(file_path)
                    print(f"PDF has {len(images)} pages")
                except Exception as e:
                    os.remove(file_path)
                    return jsonify({"error": f"Failed to convert PDF to images: {str(e)}"}), 500
                
                pages = []
                total_lines = 0
                total_words = 0
                total_chars = 0
                total_ottaksharas = 0
                total_predictions = 0
                avg_confidence_sum = 0
                avg_char_area_sum = 0
                
                for page_index, image in enumerate(images):
                    print(f"Processing page {page_index + 1} (index {page_index})")
                    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_page_{page_index}.png")
                    image.save(temp_image_path, 'PNG')
                    temp_files.append(temp_image_path)
                    
                    try:
                        segmentation_result, predicted_text = model_module.process_image_with_predictions(
                            temp_image_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std,
                            main_class_names, ott_class_names, app.config['STATIC_FOLDER'], session_id, page_index
                        )
                        print(f"Page {page_index} predicted text:\n{predicted_text}")
                        
                        save_line_and_word_images(segmentation_result, app.config['STATIC_FOLDER'], session_id, page_index)
                        predicted_text_url = save_predicted_text(predicted_text, app.config['STATIC_FOLDER'], session_id, page_index)
                        metrics, predictions = model_module.compute_metrics(segmentation_result, session_id, page_index)
                        
                        # Clear image data from response to reduce memory usage
                        for line_id in segmentation_result["lines"]:
                            line_data = segmentation_result["lines"][line_id]
                            line_data["line_img"] = None
                            if "words" in line_data:
                                for word_id in line_data["words"]:
                                    word_data = line_data["words"][word_id]
                                    word_data["word_img"] = None
                                    if "characters" in word_data:
                                        for char in word_data["characters"]:
                                            char["char_img"] = None
                                            if "ottaksharas" in char:
                                                for ott in char["ottaksharas"]:
                                                    ott["char_img"] = None
                        
                        total_lines += metrics.get('total_lines', 0)
                        total_words += metrics.get('total_words', 0)
                        total_chars += metrics.get('total_chars', 0)
                        total_ottaksharas += metrics.get('total_ottaksharas', 0)
                        total_predictions += metrics.get('total_predictions', 0)
                        avg_confidence_sum += metrics.get('avg_confidence', 0)
                        avg_char_area_sum += metrics.get('avg_char_area', 0)
                        
                        pages.append({
                            "page_index": page_index,
                            "segmentation": segmentation_result,
                            "predictions": predictions,
                            "predicted_text_url": predicted_text_url,
                            "metricsInPage": metrics
                        })
                    
                    except Exception as e:
                        print(f"Error processing page {page_index}: {e}")
                        pages.append({"page_index": page_index, "error": f"Failed to process page {page_index}: {str(e)}"})
                
                page_count = len(images)
                response_data = {
                    "pages": pages,
                    "metrics": {
                        "total_pages": page_count,
                        "total_lines": total_lines,
                        "total_words": total_words,
                        "total_chars": total_chars,
                        "total_ottaksharas": total_ottaksharas,
                        "total_predictions": total_predictions,
                        "avg_confidence": avg_confidence_sum / page_count if page_count > 0 else 0,
                        "avg_char_area": avg_char_area_sum / page_count if page_count > 0 else 0
                    },
                    "session_id": session_id,
                    "model_used": model_type
                }
                
            else:
                try:
                    segmentation_result, predicted_text = model_module.process_image_with_predictions(
                        file_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std,
                        main_class_names, ott_class_names, app.config['STATIC_FOLDER'], session_id, page_index=0
                    )
                    print(f"Image predicted text:\n{predicted_text}")
                    
                    save_line_and_word_images(segmentation_result, app.config['STATIC_FOLDER'], session_id, page_index=0)
                    predicted_text_url = save_predicted_text(predicted_text, app.config['STATIC_FOLDER'], session_id, page_index=0)
                    metrics, predictions = model_module.compute_metrics(segmentation_result, session_id, page_index=0)
                    
                    # Clear image data from response
                    for line_id in segmentation_result["lines"]:
                        line_data = segmentation_result["lines"][line_id]
                        line_data["line_img"] = None
                        if "words" in line_data:
                            for word_id in line_data["words"]:
                                word_data = line_data["words"][word_id]
                                word_data["word_img"] = None
                                if "characters" in word_data:
                                    for char in word_data["characters"]:
                                        char["char_img"] = None
                                        if "ottaksharas" in char:
                                            for ott in char["ottaksharas"]:
                                                ott["char_img"] = None
                    
                    response_data = {
                        "pages": [{
                            "page_index": 0,
                            "segmentation": segmentation_result,
                            "predictions": predictions,
                            "predicted_text_url": predicted_text_url,
                            "metricsInPage": metrics
                        }],
                        "metrics": {
                            "total_pages": 1,
                            "total_lines": metrics.get('total_lines', 0),
                            "total_words": metrics.get('total_words', 0),
                            "total_chars": metrics.get('total_chars', 0),
                            "total_ottaksharas": metrics.get('total_ottaksharas', 0),
                            "total_predictions": metrics.get('total_predictions', 0),
                            "avg_confidence": metrics.get('avg_confidence', 0),
                            "avg_char_area": metrics.get('avg_char_area', 0)
                        },
                        "session_id": session_id,
                        "model_used": model_type
                    }
                
                except Exception as e:
                    os.remove(file_path)
                    return jsonify({"error": f"Error processing image: {str(e)}"}), 500
            
            # Save response_data to JSON
            try:
                response_data_path = os.path.join(session_static_folder, 'response_data.json')
                with open(response_data_path, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                print(f"Saved response_data to {response_data_path}")
            except Exception as e:
                print(f"Warning: Could not save response_data.json: {e}")
            
            # Cleanup
            try:
                os.remove(file_path)
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except OSError as e:
                print(f"Warning: Could not remove files: {e}")
            
            print(f"Processed file with {response_data['metrics']['total_predictions']} predictions")
            print(f"Response data: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            return jsonify(response_data)
        
        return jsonify({"error": "File type not allowed. Please upload PNG, JPG, JPEG, or PDF files."}), 400
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if 'session_static_folder' in locals() and os.path.exists(session_static_folder):
                shutil.rmtree(session_static_folder)
            if 'temp_files' in locals():
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/generate_transcribed_pdf/<session_id>', methods=['GET', 'POST'])
def generate_transcribed_pdf(session_id):
    try:
        session_folder = os.path.join(app.config['STATIC_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            return jsonify({"error": "Session folder not found"}), 404
        
        # Load response_data.json to get predicted_text_url for each page
        response_data_path = os.path.join(session_folder, 'response_data.json')
        if not os.path.exists(response_data_path):
            return jsonify({"error": "Response data not found"}), 404
        
        with open(response_data_path, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
        
        text_files = []
        for page in response_data['pages']:
            if 'predicted_text_url' in page:
                text_file = page['predicted_text_url'].split('/')[-1]
                text_path = os.path.join(session_folder, text_file)
                print(f"Reading text file: {text_path}")
                if os.path.exists(text_path):
                    text_files.append(text_path)
        
        if not text_files:
            return jsonify({"error": "No transcribed text files found for this session"}), 404
        
        # Get metrics from POST request if available
        metrics = None
        if request.method == 'POST':
            try:
                data = request.get_json()
                metrics = data.get('metrics', {})
            except Exception as e:
                print(f"Error parsing JSON data: {e}")
        
        # Generate HTML content
        html_content = generate_html_content(text_files, session_id, metrics)
        
        # Create temporary HTML file
        temp_html_path = os.path.join(tempfile.gettempdir(), f"transcribed_{session_id}.html")
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF using WeasyPrint
        temp_pdf_path = os.path.join(tempfile.gettempdir(), f"transcribed_{session_id}.pdf")
        
        try:
            html_doc = HTML(filename=temp_html_path)
            css = CSS(string="""
                @page {
                    @top-center {
                        content: "OCR Transcription";
                        font-size: 10px;
                        color: #666;
                    }
                }
            """)
            html_doc.write_pdf(temp_pdf_path, stylesheets=[css])
            print(f"Generated PDF using WeasyPrint at: {temp_pdf_path}")
        except Exception as e:
            print(f"WeasyPrint error: {e}")
            try:
                html_doc = HTML(filename=temp_html_path)
                html_doc.write_pdf(temp_pdf_path)
                print(f"Generated PDF using WeasyPrint (fallback) at: {temp_pdf_path}")
            except Exception as fallback_error:
                print(f"WeasyPrint fallback also failed: {fallback_error}")
                raise fallback_error
        
        return send_file(temp_pdf_path, as_attachment=True, download_name=f"transcribed_{session_id}.pdf")
    
    except Exception as e:
        print(f"Error generating PDF for session {session_id}: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500
    finally:
        for temp_file in [temp_html_path if 'temp_html_path' in locals() else None, 
                         temp_pdf_path if 'temp_pdf_path' in locals() else None]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")

@app.route('/static/<session_id>/<path:path>')
def serve_static(session_id, path):
    try:
        static_path = os.path.join(app.config['STATIC_FOLDER'], session_id)
        if not os.path.exists(static_path):
            return jsonify({"error": "Session folder not found"}), 404
        return send_from_directory(static_path, path)
    except Exception as e:
        print(f"Error serving static file {session_id}/{path}: {e}")
        return jsonify({"error": "File not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "static_folder": app.config['STATIC_FOLDER'],
        "weasyprint_version": weasyprint.__version__
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting Kannada OCR Flask Server with WeasyPrint...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Static folder: {STATIC_FOLDER}")
    print("Folder structure: session_id/lines/, session_id/words/, session_id/chars/")
    print("Allowed file types: PNG, JPG, JPEG, PDF")
    print("Available models: densenet121_attention, botnet_resnet50, vit_base")
    print(f"WeasyPrint version: {weasyprint.__version__}")
    print("Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)