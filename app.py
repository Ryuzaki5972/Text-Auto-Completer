from flask import Flask, request, jsonify, render_template
import pandas as pd
from TextAutoCompleter import TextAutoCompleter
import os
import tempfile

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
        text = data.get('text', '')
        
        if not autocompleter.is_trained:
            return jsonify({'suggestions': [], 'error': 'Model not trained'})
        
        suggestions = autocompleter.complete_text(text, max_completions=8)
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        return jsonify({'suggestions': [], 'error': str(e)})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
        
        # Read the CSV file directly from memory (don't save to disk)
        try:
            df = pd.read_csv(file.stream)
        except Exception as csv_error:
            return jsonify({'success': False, 'error': f'Error reading CSV: {str(csv_error)}'})
        
        # Check for required column
        if 'text' not in df.columns:
            available_cols = ', '.join(df.columns.tolist())
            return jsonify({
                'success': False, 
                'error': f'CSV must contain a "text" column. Available columns: {available_cols}'
            })
        
        # Prepare training data
        texts = df['text'].dropna().astype(str).tolist()
        
        if len(texts) < 10:
            return jsonify({
                'success': False, 
                'error': f'Need at least 10 text samples to train. Found {len(texts)} samples.'
            })
        
        # Train the model
        autocompleter.train_models(texts)
        
        # Try to save the trained model (might fail in read-only environments)
        try:
            model_path = os.path.join(tempfile.gettempdir(), 'trained_model.pkl')
            autocompleter.save_models(model_path)
        except Exception as save_error:
            print(f"Warning: Could not save model to disk: {save_error}")
            # Continue anyway - model is still trained in memory
        
        return jsonify({
            'success': True, 
            'message': f'Model trained successfully on {len(texts)} text samples'
        })
    
    except Exception as e:
        print(f"Training error: {str(e)}")  # Log for debugging
        return jsonify({'success': False, 'error': f'Training failed: {str(e)}'})

@app.route('/status')
def model_status():
    return jsonify({
        'is_trained': autocompleter.is_trained,
        'unigram_size': len(autocompleter.unigram_model),
        'bigram_size': len(autocompleter.bigram_model),
        'trigram_size': len(autocompleter.trigram_model)
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

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
    app.run(host='0.0.0.0', port=port, debug=False)