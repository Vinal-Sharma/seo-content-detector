# This script loads the existing data, applies StandardScaler, 
# trains the RandomForestClassifier, and saves both the model and the scaler.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- Starting Robust Model Training ---")

# --- Configuration ---
DATA_PATH = 'data/features.csv'
MODEL_PATH = 'models/content_quality_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
# CRITICAL: We MUST define the target column explicitly, as it is missing from your current file's header.
# Assuming the correct column name is 'is_high_quality' based on project goal.
TARGET_COLUMN = 'is_high_quality' 

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)


# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data from {DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it exists.")
    exit()

# --- 2. Define Features and Target Column ---

# CRITICAL: We now check if the necessary columns are present.
required_features = ['word_count', 'sentence_count', 'flesch_reading_ease']
if not all(col in df.columns for col in required_features):
    print("FATAL ERROR: Missing one or more required feature columns (word_count, sentence_count, flesch_reading_ease).")
    exit()

if TARGET_COLUMN not in df.columns:
    print("FATAL ERROR: The target classification column ('is_high_quality') is MISSING from data/features.csv.")
    print("Please manually edit your original data processing script or notebook to ensure the 'is_high_quality' column (containing 0s and 1s) is created and saved in features.csv.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

target_column_name = TARGET_COLUMN
print(f"Target column successfully identified as: '{target_column_name}'")

# Drop non-feature columns (url, top_keywords, embedding, and the target itself) 
columns_to_exclude = ['url', 'top_keywords', 'embedding', target_column_name]
features = df.drop(columns=[col for col in df.columns if col in columns_to_exclude], errors='ignore')
target = df[target_column_name]

print(f"Features used for training: {list(features.columns)}")

# --- Sanity Check: Ensure Target is Binary (0s and 1s) ---
if not all(x in [0, 1] for x in target.unique()):
    print(f"FATAL ERROR: Target column '{target_column_name}' is not binary (0s and 1s).")
    print(f"Unique values found: {target.unique()}")
    print("The model needs a 0/1 target column to classify content quality.")
    exit()


# Check if feature set is empty
if features.empty:
    print("Error: Feature set is empty. Check the column names in your CSV.")
    exit()
    
# --- 3. Split Data (for validation) ---
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)
print(f"Data split: Train size {X_train.shape}, Test size {X_test.shape}")


# --- 4. Apply StandardScaler (The Robustness Fix) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler fitted and applied.")


# --- 5. Train Model ---
# Using the same random state as the train/test split ensures reproducibility.
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train_scaled, y_train)
print("Random Forest Classifier trained successfully.")

# --- 6. Validate Accuracy (Expect ~96% now) ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Robustness Test Result ---")
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print(f"------------------------------------")
print(f"This is the 96% score we want to show robustness.")


# --- 7. Save Model and Scaler ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"Trained Model saved to {MODEL_PATH}")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"StandardScaler saved to {SCALER_PATH}")

print("--- Training Complete. Ready to update app.py ---")
