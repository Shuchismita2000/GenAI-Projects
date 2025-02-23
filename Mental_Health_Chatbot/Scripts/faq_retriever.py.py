import pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer

# Pinecone API Key (Replace with your own)
PINECONE_API_KEY = "pcsk_6WjJ7w_Mdge2WY3AVCqN5RafTo3byPqv7ECMdRSr1SZ4oS6quNbkDb4Lc7tnE2esM9DKQe"
INDEX_NAME = "mental-health-bot"

# Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

# Load the Sentence Transformer Model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAQ Data
faq_df = pd.read_csv(r"D:\gen-ai-trial\GenAI-Projects\mental_health_bot\data\faq_3512.csv")

# Create Pinecone Index (if it doesn’t exist)
if INDEX_NAME not in [index_info["name"] for index_info in pinecone_client.list_indexes()]:
    pinecone_client.create_index(name=INDEX_NAME, dimension=384,metric="cosine")

# Connect to Pinecone Index
index = pinecone_client.Index(INDEX_NAME)

# Convert Questions into Embeddings & Store in Pinecone
for idx, row in faq_df.iterrows():
    vector = embed_model.encode(row["Context"]).tolist()  # Convert text to vector
    index.upsert([(str(idx), vector, {"Response": row["Response"]})])

print("✅ FAQ Data Successfully Uploaded to Pinecone")

def retrieve_faq(question):
    """
    Searches Pinecone for the most relevant FAQ answer.
    
    :param question: (str) The user's input question.
    :return: (str) The best-matching FAQ answer.
    """
    query_vector = embed_model.encode(question).tolist()  # Convert query to vector
    result = index.query(vector=query_vector, top_k=1, include_metadata=True)

    if result["matches"]:
        return result["matches"][0]["metadata"]["Response"]
    else:
        return "I'm sorry, I couldn't find an answer to that question."

# Example Test
if __name__ == "__main__":
    user_query = "What are the symptoms of depression?"
    answer = retrieve_faq(user_query)
    print(f"User Query: {user_query}")
    print(f"Retrieved Answer: {answer}")
