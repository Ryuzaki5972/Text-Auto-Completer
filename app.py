from flask import Flask, request, jsonify, render_template
import pandas as pd
from TextAutoCompleter import TextAutoCompleter

import os

app = Flask(__name__)

# Initialize the autocompleter
autocompleter = TextAutoCompleter()

# HTML template for the web interface

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
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        if 'text' not in df.columns:
            return jsonify({'success': False, 'error': 'CSV must contain a "text" column'})
        
        # Train the model
        texts = df['text'].dropna().tolist()
        if len(texts) < 10:
            return jsonify({'success': False, 'error': 'Need at least 10 text samples to train'})
        
        autocompleter.train_models(texts)
        
        # Save the trained model
        autocompleter.save_models('trained_model.pkl')
        
        return jsonify({'success': True, 'message': f'Model trained on {len(texts)} texts'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def model_status():
    return jsonify({
        'is_trained': autocompleter.is_trained,
        'unigram_size': len(autocompleter.unigram_model),
        'bigram_size': len(autocompleter.bigram_model),
        'trigram_size': len(autocompleter.trigram_model)
    })

if __name__ == '__main__':
    # Try to load existing model if available
    if os.path.exists('trained_model.pkl'):
        try:
            autocompleter.load_models('trained_model.pkl')
            print("Loaded existing trained model.")
        except:
            print("Could not load existing model. Will need to train a new one.")
    
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)