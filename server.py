from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import PyPDF2
import io
import magic
from docx import Document
import json
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_text_from_file(file):
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file.read(1024))
    file.seek(0)  # Reset file pointer
    
    if 'pdf' in file_type:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif 'word' in file_type or 'docx' in file_type:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:  # Assume text file
        text = file.read().decode('utf-8')
    
    return text

def analyze_text(text):
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Get word frequency
    freq_dist = FreqDist(words)
    key_topics = freq_dist.most_common(5)
    
    # Named Entity Recognition
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Calculate metadata
    metadata = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'unique_words': len(set(words)),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }
    
    # Calculate document score
    score = (
        (sentiment_scores['pos'] * 100) -
        (sentiment_scores['neg'] * 50) +
        (sentiment_scores['neu'] * 25)
    ) / 1.75
    
    return {
        'sentiment': {
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu']
        },
        'key_topics': key_topics,
        'score': min(max(score, 0), 100),
        'summary': sentences[:3],
        'metadata': metadata,
        'named_entities': entities
    }

@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'message': 'AI Document Analyzer API is running'
    })

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
            
        # Extract text from file
        text = extract_text_from_file(file)
        if not text.strip():
            return jsonify({'error': 'No text content found in file'}), 400
            
        # Analyze the text
        analysis_result = analyze_text(text)
        return jsonify(analysis_result)
        
    except Exception as e:
        app.logger.error(f'Error processing file: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
