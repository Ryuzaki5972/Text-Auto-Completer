from flask import Flask, request, jsonify, render_template
import pandas as pd
from TextAutoCompleter import TextAutoCompleter
import os
import tempfile
import traceback
import sys
import glob

app = Flask(__name__)

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

def load_chunk_files():
    """Load all chunk files from dataset folder"""
    dataset_folder = 'dataset'
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f'Dataset folder not found at {dataset_folder}')
    
    # Find all chunk files (chunk_1.csv, chunk_2.csv, etc.)
    chunk_pattern = os.path.join(dataset_folder, 'chunk_*.csv')
    chunk_files = sorted(glob.glob(chunk_pattern), key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    
    if not chunk_files:
        raise FileNotFoundError(f'No chunk files found in {dataset_folder}. Expected files like chunk_1.csv, chunk_2.csv, etc.')
    
    print(f"Found {len(chunk_files)} chunk files: {[os.path.basename(f) for f in chunk_files]}", file=sys.stderr)
    
    all_texts = []
    total_rows = 0
    
    for chunk_file in chunk_files:
        try:
            print(f"Loading {os.path.basename(chunk_file)}...", file=sys.stderr)
            df = pd.read_csv(chunk_file)
            
            # Check for required column in first chunk
            if len(all_texts) == 0 and 'text' not in df.columns:
                available_cols = ', '.join(df.columns.tolist())
                raise ValueError(f'CSV files must contain a "text" column. Available columns in {os.path.basename(chunk_file)}: {available_cols}')
            
            # Extract text data
            if 'text' in df.columns:
                chunk_texts = df['text'].dropna().astype(str).tolist()
                all_texts.extend(chunk_texts)
                total_rows += len(df)
                print(f"  - Loaded {len(chunk_texts)} texts from {os.path.basename(chunk_file)}", file=sys.stderr)
            else:
                print(f"  - Warning: {os.path.basename(chunk_file)} missing 'text' column, skipping", file=sys.stderr)
                
        except Exception as e:
            print(f"Error loading {chunk_file}: {str(e)}", file=sys.stderr)
            continue
    
    print(f"Total texts loaded: {len(all_texts)} from {len(chunk_files)} chunks", file=sys.stderr)
    return all_texts, len(chunk_files), total_rows

@app.route('/train', methods=['POST'])
def train_model():
    try:
        print("Training request received", file=sys.stderr)
        
        try:
            texts, chunk_count, total_rows = load_chunk_files()
        except Exception as load_error:
            error_msg = str(load_error)
            print(f"Error loading chunk files: {error_msg}", file=sys.stderr)
            return jsonify({'success': False, 'error': error_msg})
        
        if len(texts) < 10:
            error_msg = f'Need at least 10 text samples to train. Found {len(texts)} samples across {chunk_count} chunks.'
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
        
        success_msg = f'Model trained successfully on {len(texts)} text samples from {chunk_count} chunk files'
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
        # Check dataset files
        dataset_folder = 'dataset'
        chunk_pattern = os.path.join(dataset_folder, 'chunk_*.csv')
        chunk_files = glob.glob(chunk_pattern)
        dataset_exists = len(chunk_files) > 0
        
        return jsonify({
            'is_trained': autocompleter.is_trained,
            'unigram_size': len(autocompleter.unigram_model),
            'bigram_size': len(autocompleter.bigram_model),
            'trigram_size': len(autocompleter.trigram_model),
            'dataset_exists': dataset_exists,
            'chunk_count': len(chunk_files),
            'chunk_files': [os.path.basename(f) for f in sorted(chunk_files, key=lambda x: int(x.split('chunk_')[1].split('.')[0]))]
        })
    except Exception as e:
        print(f"Status error: {str(e)}", file=sys.stderr)
        return jsonify({
            'is_trained': False,
            'unigram_size': 0,
            'bigram_size': 0,
            'trigram_size': 0,
            'dataset_exists': False,
            'chunk_count': 0,
            'chunk_files': [],
            'error': str(e)
        })

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
    
    # Auto-train on startup if chunk files exist and model not trained
    if not autocompleter.is_trained:
        try:
            print("Auto-training model on startup...")
            texts, chunk_count, total_rows = load_chunk_files()
            if len(texts) >= 10:
                autocompleter.train_models(texts)
                print(f"Auto-trained model with {len(texts)} samples from {chunk_count} chunks")
            else:
                print(f"Not enough samples for auto-training: {len(texts)}")
        except Exception as e:
            print(f"Auto-training failed: {e}")
    
    print("Starting Flask application...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)