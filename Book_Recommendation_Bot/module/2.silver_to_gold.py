"""
This script processes the Silver Layer dataset and enhances data quality by:
✅ Fixing malformed URLs
✅ Normalizing genre categories
✅ Removing short/incomplete book descriptions
✅ Truncating long text for embedding compatibility
✅ Creating a unified text field for better embeddings

Output: A refined Gold Layer dataset ready for AI-powered recommendations.
"""

import pandas as pd
import re
from fuzzywuzzy import process
from transformers import AutoTokenizer

# ---------------------- CONFIGURATION ----------------------

# Define file paths for input (Silver Layer) and output (Gold Layer)
SILVER_LAYER_PATH = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\silver.csv"
GOLD_LAYER_PATH = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\gold.csv"

# Load the Silver Layer dataset
df = pd.read_csv(SILVER_LAYER_PATH)

# ---------------------- DATA CLEANING ----------------------

# 1️⃣ Fix malformed URLs (Ensure all URLs start with 'https://')
df['thumbnail'] = df['thumbnail'].apply(lambda x: "https://" + x if not x.startswith("http") else x)

# 2️⃣ Normalize genre categories (Map inconsistent genre names to standard ones)
GENRE_MAPPING = {
    "Sci-Fi": "Science Fiction",
    "YA": "Young Adult",
    "Self-Help": "Personal Development",
    "Christian Life": "Religion",
    "Detective Mystery Stories": "Mystery",
    "American Fiction": "Fiction"
}

# Function to normalize genre names using fuzzy matching
def normalize_genre(genre):
    if pd.isna(genre):  # Handle missing genre values
        return "Unknown"
    best_match = process.extractOne(genre, GENRE_MAPPING.keys(), score_cutoff=80)
    return GENRE_MAPPING[best_match[0]] if best_match else genre

df['categories'] = df['categories'].apply(normalize_genre)

# 3️⃣ Remove books with extremely short descriptions (<10 words)
df = df[df['description'].apply(lambda x: len(str(x).split()) >= 10)]

# 4️⃣ Ensure text length is within embedding model limits (Max: 512 tokens)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def truncate_text(text, max_tokens=512):
    """Truncates text to ensure it does not exceed the model's max token length."""
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

df['description'] = df['description'].apply(lambda x: truncate_text(str(x)))

# 5️⃣ Concatenate key book details into a unified text representation for embeddings
df['book_text'] = df.apply(lambda row: f"{row['title']} by {row['authors']}. Genre: {row['categories']}. Summary: {row['description']}.", axis=1)

# ---------------------- EXPORT CLEANED DATA ----------------------

# Save the processed dataset as Gold Layer CSV
df.to_csv(GOLD_LAYER_PATH, index=False)

print(f"✅ Data cleaning completed. Gold Layer dataset saved at: {GOLD_LAYER_PATH}")
