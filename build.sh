#!/usr/bin/env bash
# Exit on error
set -e

# Upgrade pip first
python -m pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
