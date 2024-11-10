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
CORS(app, resources={
    r"/*": {
        "origins": ["https://verdant-youtiao-abef68.netlify.app", "http://localhost:5173"],
        "methods": ["POST", "OPTIONS", "GET"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_file(file_bytes, mime_type):
    """Extract text from various file formats."""
    try:
        if 'pdf' in mime_type:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ' '.join(page.extract_text() for page in pdf_reader.pages)
        elif 'word' in mime_type or 'docx' in mime_type:
            doc = Document(io.BytesIO(file_bytes))
            text = ' '.join(paragraph.text for paragraph in doc.paragraphs)
        elif 'text' in mime_type:
            text = file_bytes.decode('utf-8')
        else:
            raise Exception('Unsupported file format')
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def analyze_text(text):
    """Perform comprehensive NLP analysis on text."""
    try:
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        # spaCy analysis
        doc = nlp(text)
        
        # Sentiment Analysis using NLTK's VADER
        sia = SentimentIntensityAnalyzer()
        sentences = sent_tokenize(text)
        sentiment_scores = [sia.polarity_scores(sentence) for sentence in sentences]
        
        overall_sentiment = {
            'positive': sum(score['pos'] for score in sentiment_scores) / len(sentiment_scores),
            'negative': sum(score['neg'] for score in sentiment_scores) / len(sentiment_scores),
            'neutral': sum(score['neu'] for score in sentiment_scores) / len(sentiment_scores)
        }
        
        # Key topics extraction using spaCy
        topics = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']:
                topics.append((ent.text, ent.label_))
        
        # Get most frequent noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        freq_dist = FreqDist(noun_phrases)
        key_topics = freq_dist.most_common(5)
        
        # Generate summary using spaCy
        sentences = [sent.text.strip() for sent in doc.sents]
        summary = sentences[:3]  # Take first 3 sentences as summary
        
        # Calculate document complexity score
        words = word_tokenize(text)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        
        # Calculate readability score (simplified Flesch-Kincaid)
        total_sentences = len(sentences)
        total_words = len(words)
        
        if total_words == 0:
            raise Exception("Document is empty")
            
        score = (
            (overall_sentiment['positive'] * 100) +
            (unique_words / total_words * 50) +
            (min(avg_word_length, 10) * 5)
        ) / 1.75
        
        return {
            'sentiment': overall_sentiment,
            'key_topics': key_topics,
            'named_entities': topics[:5],
            'score': min(max(score, 0), 100),
            'summary': summary,
            'metadata': {
                'word_count': total_words,
                'sentence_count': total_sentences,
                'unique_words': unique_words,
                'avg_word_length': round(avg_word_length, 2)
            }
        }
    except Exception as e:
        raise Exception(f"Error analyzing text: {str(e)}")

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        file = request.files['file']
        file_bytes = file.read()
        
        # Detect mime type
        mime_type = magic.from_buffer(file_bytes, mime=True)
        
        # Extract text
        text = extract_text_from_file(file_bytes, mime_type)
        if not text.strip():
            return jsonify({'error': 'No text content found in document'}), 400
        
        # Analyze text
        analysis = analyze_text(text)
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
