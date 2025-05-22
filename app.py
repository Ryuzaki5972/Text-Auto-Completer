from flask import Flask, request, jsonify, render_template
import pandas as pd
from TextAutoCompleter import TextAutoCompleter
import os
import tempfile
import traceback
import sys

app = Flask(__name__)

# Configure file uploads for large files
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_TIMEOUT'] = 600  # 10 minutes timeout
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use temp directory

# Initialize the autocompleter
autocompleter = TextAutoCompleter()

@app.route('/')
def index():
    return render_template("index.html", autocompleter=autocompleter)

@app.route('/complete', methods=['POST'])
def complete_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'suggestions': [], 'error': 'No data provided'})
            
        text = data.get('text', '')
        
        if not autocompleter.is_trained:
            return jsonify({'suggestions': [], 'error': 'Model not trained'})
        
        suggestions = autocompleter.complete_text(text, max_completions=8)
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        print(f"Error in complete_text: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'suggestions': [], 'error': str(e)})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        print("Training request received", file=sys.stderr)
        
        # Check if file is in request
        if 'file' not in request.files:
            print("No file in request", file=sys.stderr)
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        print(f"File received: {file.filename}", file=sys.stderr)
        
        # Check if file was selected
        if file.filename == '':
            print("Empty filename", file=sys.stderr)
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            print(f"Invalid file extension: {file.filename}", file=sys.stderr)
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
        
        # Read the CSV file directly from memory
        try:
            print("Reading CSV file...", file=sys.stderr)
            # Reset file pointer to beginning
            file.stream.seek(0)
            df = pd.read_csv(file.stream)
            print(f"CSV loaded with shape: {df.shape}", file=sys.stderr)
            print(f"Columns: {df.columns.tolist()}", file=sys.stderr)
        except Exception as csv_error:
            print(f"CSV reading error: {str(csv_error)}", file=sys.stderr)
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Error reading CSV: {str(csv_error)}'})
        
        # Check for required column
        if 'text' not in df.columns:
            available_cols = ', '.join(df.columns.tolist())
            error_msg = f'CSV must contain a "text" column. Available columns: {available_cols}'
            print(error_msg, file=sys.stderr)
            return jsonify({'success': False, 'error': error_msg})
        
        # Prepare training data
        print("Preparing training data...", file=sys.stderr)
        texts = df['text'].dropna().astype(str).tolist()
        print(f"Found {len(texts)} text samples", file=sys.stderr)
        
        if len(texts) < 10:
            error_msg = f'Need at least 10 text samples to train. Found {len(texts)} samples.'
            print(error_msg, file=sys.stderr)
            return jsonify({'success': False, 'error': error_msg})
        
        # Train the model
        print("Starting model training...", file=sys.stderr)
        autocompleter.train_models(texts)
        print("Model training completed", file=sys.stderr)
        
        # Try to save the trained model
        try:
            model_path = os.path.join(tempfile.gettempdir(), 'trained_model.pkl')
            autocompleter.save_models(model_path)
            print(f"Model saved to {model_path}", file=sys.stderr)
        except Exception as save_error:
            print(f"Warning: Could not save model to disk: {save_error}", file=sys.stderr)
        
        success_msg = f'Model trained successfully on {len(texts)} text samples'
        print(success_msg, file=sys.stderr)
        return jsonify({'success': True, 'message': success_msg})
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"Training error: {error_msg}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})

@app.route('/status')
def model_status():
    try:
        return jsonify({
            'is_trained': autocompleter.is_trained,
            'unigram_size': len(autocompleter.unigram_model),
            'bigram_size': len(autocompleter.bigram_model),
            'trigram_size': len(autocompleter.trigram_model)
        })
    except Exception as e:
        print(f"Status error: {str(e)}", file=sys.stderr)
        return jsonify({
            'is_trained': False,
            'unigram_size': 0,
            'bigram_size': 0,
            'trigram_size': 0,
            'error': str(e)
        })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    print("File too large error", file=sys.stderr)
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {str(e)}", file=sys.stderr)
    traceback.print_exc()
    return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.before_request
def log_request_info():
    print(f"Request: {request.method} {request.url}", file=sys.stderr)

if __name__ == '__main__':
    # Try to load existing model if available
    model_paths = [
        'trained_model.pkl',
        os.path.join(tempfile.gettempdir(), 'trained_model.pkl')
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                autocompleter.load_models(model_path)
                print(f"Loaded existing trained model from {model_path}")
                break
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}")
    
    print("Starting Flask application...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  # Enable debug mode