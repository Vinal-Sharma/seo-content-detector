import streamlit as st
import pickle
import re
import nltk
import textstat
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler # Added for professional robustness discussion

# Set the page configuration for a professional look
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")


# ------------------------------------------------------------------
# 1. LOAD YOUR SAVED MODEL AND SCALER
# ------------------------------------------------------------------
# Define model path
MODELS_DIR = 'models'
MODEL_FILE = os.path.join(MODELS_DIR, 'content_quality_model_HACK.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl') # Assuming you save the scaler too

# --- NLTK Data Fix for Streamlit ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# --- End of fix ---


# Load Model
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE}' not found.")
    st.error("Please make sure the model file is in the same folder as app.py or models/.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
    
# NOTE: If you trained your final model with the StandardScaler, you should save and load it here.
# If you didn't, the line below will cause an error, but we'll include the scaler object in 
# the prediction step to allow you to talk about it (as discussed for the interview).
# We will assume your model is still the basic one for now, as the model file is not available.

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
# 3. BUILD THE STREAMLIT APP INTERFACE (IMPROVED)
# ------------------------------------------------------------------

# --- Sidebar for instructions and professionalism ---
with st.sidebar:
    st.header("App Instructions")
    st.info("Paste the raw HTML of a web page into the box below. Our model instantly classifies it as 'High' or 'Low' quality based on engineered SEO metrics.")
    st.subheader("Model Pipeline:")
    st.markdown("- **Model:** `RandomForestClassifier`")
    st.markdown("- **Features:** Word Count, Sentence Count, Flesch Readability.")
    st.markdown("- **Goal:** Predict content that is 'Thin' or 'Low Quality'.")

# --- Main App Title ---
st.title("🤖 SEO Content Quality Analyzer")
st.write("This app demonstrates the deployment of a Machine Learning content analysis pipeline for real-time assessment.")
st.markdown("---")

# --- Sample HTML for easy testing ---
high_quality_sample = """
<html><body>
<h1>A Comprehensive Guide to Modern SEO Techniques</h1>
<p>A successful content strategy is the backbone of any modern digital marketing effort. It ensures that every piece of writing, every blog post, and every social media update serves a clear business objective. Without a strategy, content can quickly become disorganized and fail to engage the target audience effectively.</p>
<p>Our research shows that articles over 500 words tend to rank better in search engine results. This is not because search engines favor length, but because longer articles typically cover a topic with more depth and authority. When creating your next article, focus on providing comprehensive value to the reader. Use clear headings and short sentences to improve the flow and readability.</p>
</body></html>
"""
low_quality_sample = """
<html><body>
<h1>Test Article</h1>
<p>Hello. This is a simple test article to check the content quality analyzer. Not much text here.</p>
</body></html>
"""

# Text area input
raw_html = st.text_area(
    "1. Paste Raw HTML Content Here:", 
    height=250, 
    key="html_input", # Added key for state management
    placeholder="<html><body><p>Your content goes here...</p></body></html>"
)

# Sample loading buttons
col_sample_1, col_sample_2 = st.columns(2)

with col_sample_1:
    if st.button('Load High-Quality Sample'):
        st.session_state.html_input = high_quality_sample
        st.experimental_rerun()
with col_sample_2:
    if st.button('Load Low-Quality Sample'):
        st.session_state.html_input = low_quality_sample
        st.experimental_rerun()


st.markdown("---")

# Create a button to run the analysis
if st.button("2. Analyze Content", type="primary"):
    if not st.session_state.html_input:
        st.warning("Please paste some HTML content to analyze.")
    else:
        with st.spinner("Analyzing..."):
            
            # --- Pipeline execution ---
            title, body_text, word_count = parse_html(st.session_state.html_input)
            
            if word_count == 0:
                st.error("### Prediction: LOW QUALITY (No text found in HTML)")
            else:
                _, sentence_count, flesch_score = extract_text_features(body_text)
                
                # --- PREDICTION STEP (Features must match training features!) ---
                # NOTE: For a robust discussion, you must talk about scaling. 
                # If your model was trained *without* a scaler, you skip the scaling step here.
                # If you implement the scaler fix in your notebook, you must load and use the scaler here.
                
                features_array = np.array([[word_count, sentence_count, flesch_score]])
                
                # If you fixed your notebook with the StandardScaler, add this line (assuming you save the scaler):
                # features_array_scaled = scaler.transform(features_array) 
                
                prediction = model.predict(features_array)
                
                # --- Result Display ---
                st.subheader("✅ Analysis Results and Prediction")
                
                # Use columns for neat metric display
                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", f"{word_count:,}")
                col2.metric("Sentence Count", f"{sentence_count:,}")
                col3.metric("Flesch Readability", f"{flesch_score:.2f}")

                st.markdown("---")
                
                # Final Prediction Output
                if prediction[0] == True:
                    st.error("### 🔴 Prediction: LOW QUALITY")
                    st.write("This content is predicted as 'Low Quality' because its combination of length and readability fails the quality threshold.")
                else:
                    st.success("### 🟢 Prediction: HIGH QUALITY")
                    st.write("This content is predicted as 'High Quality' based on its robust word count and accessible readability score.")

                st.markdown("---")
                st.text_area("Clean Extracted Text (for Review)", body_text, height=150, disabled=True)
