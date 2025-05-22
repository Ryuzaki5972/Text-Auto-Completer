from flask import Flask, request, jsonify, render_template
import os
import sys
import traceback

app = Flask(__name__)

# Initialize the autocompleter as None - will be loaded from pickle file
autocompleter = None

def load_pretrained_model():
    """Load the pre-trained model from trained_model.pkl"""
    global autocompleter
    
    if autocompleter is not None:
        return autocompleter
        
    try:
        # Import here to avoid startup issues
        from TextAutoCompleter import TextAutoCompleter
        
        # Look for trained_model.pkl in current directory
        model_path = 'trained_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found in current directory")
            print(f"Current directory contents: {os.listdir('.')}")
            return None
            
        print(f"Loading pre-trained model from {model_path}...")
        autocompleter = TextAutoCompleter()
        autocompleter.load_models(model_path)
        print("✅ Pre-trained model loaded successfully!")
        print(f"Model stats - Unigrams: {len(autocompleter.unigram_model)}, Bigrams: {len(autocompleter.bigram_model)}, Trigrams: {len(autocompleter.trigram_model)}")
        
        return autocompleter
        
    except Exception as e:
        print(f"❌ Failed to load pre-trained model: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    completer = load_pretrained_model()
    return render_template("index.html", autocompleter=completer)

@app.route('/complete', methods=['POST'])
def complete_text():
    try:
        completer = load_pretrained_model()
        
        if completer is None:
            return jsonify({'suggestions': [], 'error': 'Pre-trained model not available'})
            
        data = request.get_json()
        if not data:
            return jsonify({'suggestions': [], 'error': 'No data provided'})
            
        text = data.get('text', '')
        
        if not completer.is_trained:
            return jsonify({'suggestions': [], 'error': 'Model not properly loaded'})
        
        suggestions = completer.complete_text(text, max_completions=8)
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        print(f"Error in complete_text: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'suggestions': [], 'error': str(e)})

@app.route('/status')
def model_status():
    try:
        completer = load_pretrained_model()
        
        model_path = 'trained_model.pkl'
        model_exists = os.path.exists(model_path)
        
        if completer and completer.is_trained:
            return jsonify({
                'is_trained': True,
                'model_source': 'Pre-trained model (trained_model.pkl)',
                'unigram_size': len(completer.unigram_model),
                'bigram_size': len(completer.bigram_model),
                'trigram_size': len(completer.trigram_model),
                'model_file_exists': model_exists,
                'training_method': 'N-gram models with interpolation smoothing'
            })
        else:
            return jsonify({
                'is_trained': False,
                'model_source': 'trained_model.pkl not found or failed to load',
                'unigram_size': 0,
                'bigram_size': 0,
                'trigram_size': 0,
                'model_file_exists': model_exists,
                'error': 'Pre-trained model not available'
            })
            
    except Exception as e:
        print(f"Status error: {str(e)}", file=sys.stderr)
        return jsonify({
            'is_trained': False,
            'model_source': 'Error loading model',
            'unigram_size': 0,
            'bigram_size': 0,
            'trigram_size': 0,
            'model_file_exists': False,
            'error': str(e)
        })

@app.route('/model-info')
def model_info():
    """Endpoint to get information about how the model was trained"""
    return jsonify({
        'training_method': 'N-gram Language Models with Linear Interpolation',
        'model_type': 'Unigram + Bigram + Trigram with smoothing',
        'smoothing': 'Laplace (Add-1) Smoothing + Linear Interpolation',
        'preprocessing': [
            'Text cleaning (remove punctuation, non-ASCII)',
            'Stopword removal',
            'Lowercasing',
            'Number normalization (<NUM> tokens)',
            'Contraction expansion',
            'Lemmatization using WordNet',
            'Tokenization'
        ],
        'interpolation_weights': {
            'trigram_lambda': 0.749,
            'bigram_lambda': 0.010, 
            'unigram_lambda': 0.241
        },
        'training_corpus': 'Text dataset processed into n-grams',
        'vocabulary_filtering': 'Minimum threshold filtering for rare n-grams',
        'model_size': 'Depends on training corpus size and vocabulary'
    })

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {str(e)}", file=sys.stderr)
    traceback.print_exc()
    return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.before_request
def log_request_info():
    if request.endpoint != 'static':
        print(f"Request: {request.method} {request.url}", file=sys.stderr)

# Health check endpoint for deployment
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': autocompleter is not None and (autocompleter.is_trained if hasattr(autocompleter, 'is_trained') else False)
    })

if __name__ == '__main__':
    print("🚀 Starting Text Auto Completer with Pre-trained Model...")
    
    # Load model on startup to check if it works
    completer = load_pretrained_model()
    if completer:
        print("✅ Model check passed - ready to serve requests")
    else:
        print("⚠️  Model not loaded - app will run but autocomplete won't work")
        print("📋 Make sure 'trained_model.pkl' is in the same directory as app.py")
    
    # Get port from environment (important for Render)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)