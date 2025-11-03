import streamlit as st
import pickle
import re
import nltk
import textstat
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------
# 1. LOAD YOUR SAVED MODEL
# ------------------------------------------------------------------
# Define model path
# We place this in the main project folder.
# Define model path
MODELS_DIR = 'models'
MODEL_FILE = os.path.join(MODELS_DIR, 'content_quality_model_HACK.pkl')

# --- This is a fix for the NLTK data on Streamlit ---
# Try to find the data. If not, download it.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# --- End of fix ---


try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Model file 'content_quality_model_HACK.pkl' not found.")
    st.error("Please make sure the model file is in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ------------------------------------------------------------------
# 2. PASTE IN YOUR HELPER FUNCTIONS
# ------------------------------------------------------------------
def parse_html(html_content):
    """Parses HTML content to extract title and clean body text."""
    try:
        if pd.isna(html_content) or not isinstance(html_content, str) or html_content == 'No HTML Content':
            return "No Title", "No Content", 0
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else "No Title"
        title = re.sub(r'\s+', ' ', title).strip()
        if soup.body:
            for script in soup(["script", "style"]):
                script.extract()
            body_text = soup.body.get_text(separator=' ', strip=True)
        else:
            body_text = "No Content"
        if not body_text.strip():
             body_text = "No Content"
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        word_count = len(body_text.split())
        return title, body_text, word_count
    except Exception as e:
        return "Error Title", "Error Content", 0

def extract_text_features(text):
    """Calculates sentence count and Flesch reading ease."""
    if not isinstance(text, str) or not text.strip():
        return "", 0, 100.0 
    
    clean_text = text.lower()
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    sentence_count = 0
    flesch_score = 100.0
    try:
        sentences = nltk.sent_tokenize(clean_text)
        sentence_count = len(sentences)
    except Exception as e:
        sentence_count = 0
    try:
        if len(clean_text.split()) < 10:
            flesch_score = 100.0
        else:
            flesch_score = textstat.flesch_reading_ease(clean_text)
    except Exception as e:
        flesch_score = 0.0
    return clean_text, sentence_count, flesch_score

# ------------------------------------------------------------------
# 3. BUILD THE STREAMLIT APP INTERFACE
# ------------------------------------------------------------------
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
st.title("🤖 SEO Content Quality Analyzer")
st.write("This app uses a Machine Learning model (RandomForestClassifier) to predict if content is 'High Quality' or 'Low Quality' based on its text features (word count, readability, etc.).")

# Create a text area for user input
# Notice there is NO key='MyKey' here
raw_html = st.text_area("Paste Raw HTML Content Here", height=300, 
                        placeholder="<html><body><p>This is my article...</p></body></html>")

# Create a button to run the analysis
if st.button("Analyze Content"):
    if not raw_html:
        st.warning("Please paste some HTML content to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Step 1: Parse the HTML
            title, body_text, word_count = parse_html(raw_html)
            
            if word_count == 0:
                st.error("Prediction: LOW QUALITY (No text found in HTML)")
            else:
                # Step 2: Extract text features
                _, sentence_count, flesch_score = extract_text_features(body_text)
                
                # Step 3: Prepare features for the model
                features_array = np.array([[word_count, sentence_count, flesch_score]])
                
                # Step 4: Make a prediction!
                # (We don't need a scaler for the RandomForest model)
                prediction = model.predict(features_array)
                
                # Step 5: Show the result
                st.subheader("Analysis Results:")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", word_count)
                col2.metric("Sentence Count", sentence_count)
                col3.metric("Flesch Readability", f"{flesch_score:.2f}")

                # Remember: True == Low Quality, False == High Quality
                if prediction[0] == True:
                    st.error("### Prediction: LOW QUALITY")
                    st.write("This content is predicted as 'Low Quality' because it likely has a low word count or very poor readability.")
                else:
                    st.success("### Prediction: HIGH QUALITY")
                    st.write("This content is predicted as 'High Quality' based on its word count and readability scores.")
                    