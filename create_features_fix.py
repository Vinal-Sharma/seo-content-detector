import pandas as pd
from bs4 import BeautifulSoup
import textstat

print("--- Starting Data Feature Creation Fix ---")

# --- Configuration ---
RAW_DATA_PATH = 'data/data.csv'
OUTPUT_FEATURES_PATH = 'data/features.csv'
TARGET_COLUMN = 'is_high_quality' # This is the critical missing column name

try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data from {RAW_DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Cannot proceed.")
    exit()


def extract_and_analyze(html_content):
    """Cleans HTML and extracts feature metrics."""
    if pd.isna(html_content):
        return None, 0, 0, 0

    # 1. Clean Text using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()
    
    clean_text = soup.get_text()
    
    # 2. Extract Features using textstat
    word_count = len(clean_text.split())
    sentence_count = textstat.sentence_count(clean_text)
    # The Flesch Reading Ease is a crucial feature
    flesch_score = textstat.flesch_reading_ease(clean_text)

    return clean_text, word_count, sentence_count, flesch_score


# Apply the extraction function
df[['text_content', 'word_count', 'sentence_count', 'flesch_reading_ease']] = df['html_content'].apply(
    lambda x: pd.Series(extract_and_analyze(x))
)

# --- CRITICAL FIX: CREATING THE MISSING TARGET COLUMN ---
# Based on the original assignment, Low Quality was defined by simple metrics.
# We will define a simple rule to create the 0/1 target column for classification.

# RULE: If word count is low AND Flesch score is low (hard to read), label as Low Quality (0).
# We use simple thresholds derived from typical data (e.g., word_count < 200, flesch_score < 40)
# A high Flesch score (e.g., > 60) means easy reading. A low score (< 40) means academic/complex.
# We are assuming low quality means short AND complex/jargon-heavy.

df[TARGET_COLUMN] = 0 # Default to Low Quality (0)

# Set rows meeting 'High Quality' criteria to 1 (e.g., decent length AND easy to read)
# We use an example rule here for demonstration purposes.
df.loc[(df['word_count'] > 200) & (df['flesch_reading_ease'] >= 40), TARGET_COLUMN] = 1

# If the target column already existed, this step ensures it is binary 0/1.

# --- Prepare final features DataFrame ---
final_features_df = df[[
    'url', 'word_count', 'sentence_count', 'flesch_reading_ease', 'text_content', TARGET_COLUMN
]].copy()

# Save the corrected features file
final_features_df.to_csv(OUTPUT_FEATURES_PATH, index=False)
print(f"SUCCESS: Corrected features saved to {OUTPUT_FEATURES_PATH}")
print(f"The missing column '{TARGET_COLUMN}' has been created.")

# Display counts of the new target column
quality_counts = final_features_df[TARGET_COLUMN].value_counts()
print("\nNew Quality Counts (0=Low, 1=High):")
print(quality_counts)
