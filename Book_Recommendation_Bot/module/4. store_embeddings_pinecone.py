"""
Store Book Embeddings in Pinecone for Fast Similarity Search

This script:
✅ Loads the Gold Layer dataset with embeddings
✅ Connects to Pinecone
✅ Creates a Pinecone Index
✅ Uploads book embeddings for similarity search
✅ Performs a test query
"""

import pandas as pd
import pinecone
import os
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------------- CONFIGURATION ----------------------

# Define file paths
EMBEDDED_DATA_PATH = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\gold_with_embeddings.csv"

# Pinecone API credentials
PINECONE_API_KEY = "pcsk_6WjJ7w_Mdge2WY3AVCqN5RafTo3byPqv7ECMdRSr1SZ4oS6quNbkDb4Lc7tnE2esM9DKQe"
PINECONE_ENV = "us-east-1"  # Your selected region
INDEX_NAME = "book-recommendations"

# ---------------------- LOAD EMBEDDINGS ----------------------

# Load dataset with embeddings
df = pd.read_csv(EMBEDDED_DATA_PATH)

# Convert stringified embeddings back to lists
import ast
df['embeddings'] = df['embeddings'].apply(ast.literal_eval)

# ---------------------- INITIALIZE PINECONE ----------------------

# Initialize Pinecone client
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, otherwise create it
if INDEX_NAME not in [index_info['name'] for index_info in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=384,  # Matching all-MiniLM-L6-v2 model
        metric="cosine"
    )

# Connect to the Pinecone index
index = pinecone_client.Index(INDEX_NAME)

# ---------------------- UPLOAD EMBEDDINGS TO PINECONE ----------------------

# Prepare data for upsert (Pinecone requires tuples: (id, vector, metadata))
vectors = []
for i, row in df.iterrows():
    vectors.append((
        str(row['isbn13']),  # Unique ID
        row['embeddings'],  # Embedding vector
        {"title": row["title"], "author": row["authors"], "category": row["categories"]}  # Metadata
    ))

# Upload embeddings in batches
batch_size = 100  # Adjust batch size based on your dataset
for i in range(0, len(vectors), batch_size):
    index.upsert(vectors=vectors[i:i+batch_size])

print("✅ Embeddings successfully stored in Pinecone!")

# ---------------------- TEST QUERY: RETRIEVE SIMILAR BOOKS ----------------------

# Load HuggingFace Embeddings model for querying
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example query
query_text = "A mystery novel with a detective solving a crime"
query_embedding = hf_embeddings.embed_query(query_text)

# Search in Pinecone
search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

# Print recommended books
print("\n🔍 Top 5 Recommended Books:")
for match in search_results["matches"]:
    print(f"📖 {match['metadata']['title']} by {match['metadata']['author']} (Category: {match['metadata']['category']}) - Score: {match['score']:.2f}")



