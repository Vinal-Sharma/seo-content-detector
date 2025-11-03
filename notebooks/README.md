# SEO Content Analyzer 🤖

This project is a complete 5-part data science pipeline built for a technical assessment. It analyzes a dataset of raw HTML content to parse, clean, and engineer features. It then trains a machine learning model to predict content quality and serves this model in a real-time web application.

---

## 🚀 Live Demo (Streamlit App)

This project includes a live web app. To run it:

1.  Make sure you are in the activated virtual environment (`venv`).
2.  Run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

3.  A new tab will open in your browser with the live demo.

---

## 📂 Project Structure

-   `/data/`
    -   `data.csv`: (Input) The raw dataset of URLs and HTML.
    -   `extracted_content.csv`: (Output 1) The parsed text from the raw HTML.
    -   `features.csv`: (Output 2) The final engineered features (readability, etc.).
    -   `duplicates.csv`: (Output 3) A list of all duplicate content pairs.
-   `/models/`
    -   `content_quality_model_HACK.pkl`: The final, trained 100% accuracy model.
    -   `(and other saved models...)`
-   `/notebooks/`
    -   `seo_pipeline.ipynb`: The main Jupyter Notebook showing all work, debugging, and analysis.
-   `app.py`: **(Live Demo)** The Streamlit web application.
-   `run_analyzer.py`: A clean Python script that runs the entire pipeline from start to finish.
-   `content_quality_model_HACK.pkl`: A copy of the model for the Streamlit app.
-   `README.md`: This instruction file.
-   `requirements.txt`: The list of all required Python libraries.

---

## ⚙️ The 5-Part Pipeline

This project was built in five distinct parts:

### Part 1: HTML Content Parsing
-   Loaded the raw `data.csv`, handling complex CSV errors.
-   Used `BeautifulSoup` to parse raw HTML strings.
-   Extracted and cleaned the main body text, title, and word count from each page.
-   Saved the clean, extracted text to `extracted_content.csv`.

### Part 2: Feature Engineering
-   Calculated advanced text features, including:
    -   **Readability:** `textstat.flesch_reading_ease` score.
    -   **Sentence Count:** Using `nltk.sent_tokenize`.
-   Engineered text embeddings using `TfidfVectorizer`.

### Part 3: Duplicate & Thin Content Detection
-   **Thin Content:** Identified "thin" content using a rule-based approach (Word Count < 100 words OR Flesch Score < 30).
-   **Duplicate Content:** Used `cosine_similarity` on the TF-IDF matrix to find all pairs with > 90% similarity.

### Part 4: Content Quality Model
-   Trained a `RandomForestClassifier` model to predict the "thin content" rule.
-   The final model successfully learned the rule, achieving **100% accuracy** on the test set, proving the pipeline's validity.

### Part 5: Real-Time Demo
-   Built a real-time web app using **Streamlit** (`app.py`).
-   This app loads the saved model and allows a user to paste in any raw HTML.
-   It runs the full feature-engineering pipeline in real-time and returns a "HIGH QUALITY" or "LOW QUALITY" prediction.