"""
Generate Book Embeddings using LangChain's Hugging Face Model

This script uses LangChain's Hugging Face integration with the `all-MiniLM-L6-v2`
model to generate numerical embeddings for books.

"""

import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------------- CONFIGURATION ----------------------

# Define file paths
GOLD_LAYER_PATH = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\gold.csv"
EMBEDDED_DATA_PATH = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\gold_with_embeddings.csv"

# Load the cleaned dataset
df = pd.read_csv(GOLD_LAYER_PATH)

# ---------------------- LOAD EMBEDDING MODEL ----------------------

# Use LangChain's Hugging Face model for embeddings
print("Loading Hugging Face Embedding Model...")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------- GENERATE EMBEDDINGS ----------------------

# Generate embeddings for 'book_text' column
print("Generating embeddings...")
df['embeddings'] = df['book_text'].apply(lambda text: hf_embeddings.embed_query(text))

# Convert embeddings to a list format (easier to store in CSV)
df['embeddings'] = df['embeddings'].apply(lambda emb: np.array(emb).tolist())

# Save the dataset with embeddings
df.to_csv(EMBEDDED_DATA_PATH, index=False)

print(f"✅ Embeddings generated and saved at: {EMBEDDED_DATA_PATH}")
