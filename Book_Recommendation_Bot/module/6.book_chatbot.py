# ==============================================
# 📚 AI BOOK RECOMMENDATION CHATBOT (CONVERSATIONAL)
# ==============================================

# Import necessary libraries
import streamlit as st  # For chatbot UI
import pinecone  # For vector search
from sentence_transformers import SentenceTransformer  # For embeddings
import ollama  # For AI-generated responses

# -----------------------------------------------
# 🔹 STEP 1: INITIALIZE PINECONE CONNECTION
# -----------------------------------------------

# Pinecone API credentials
PINECONE_API_KEY = "pcsk_6WjJ7w_Mdge2WY3AVCqN5RafTo3byPqv7ECMdRSr1SZ4oS6quNbkDb4Lc7tnE2esM9DKQe"
PINECONE_ENV = "us-east-1"  # Your selected region

# Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Connect to your Pinecone index
index_name = "book-recommendations"
index = pinecone_client.Index(index_name)

# -----------------------------------------------
# 🔹 STEP 2: LOAD THE EMBEDDING MODEL
# -----------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_query_embedding(query):
    """
    Converts a user query into an embedding.
    """
    return embedding_model.encode(query).tolist()

# -----------------------------------------------
# 🔹 STEP 3: SEARCH PINECONE FOR SIMILAR BOOKS
# -----------------------------------------------
def retrieve_similar_books(user_query, top_k=5):
    """
    Searches Pinecone for the most similar book recommendations.
    """
    query_embedding = get_query_embedding(user_query)
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    recommended_books = []
    for match in search_results["matches"]:
        recommended_books.append({
            "title": match["metadata"].get("title", "Unknown Title"),
            "author": match["metadata"].get("author", "Unknown Author"),
            "genre": match["metadata"].get("category", "Unknown Category"),
            "score": match["score"]
        })

    return recommended_books

# -----------------------------------------------
# 🔹 STEP 4: FORMAT RESULTS & GENERATE RESPONSE USING OLLAMA
# -----------------------------------------------
def generate_recommendation_response(user_query, recommended_books):
    """
    Uses Ollama to generate a conversational response based on book recommendations.
    """
    book_list = "\n".join([
        f"📖 {book['title']} by {book['author']} (Genre: {book['genre']}) - Score: {book['score']:.2f}"
        for book in recommended_books
    ])

    prompt = f"""
    You are an AI book expert helping a user find the best books.

    User Query: "{user_query}"

    Here are some recommended books:

    {book_list}

    Please generate a friendly, engaging response summarizing these book recommendations.
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# -----------------------------------------------
# 🔹 STEP 5: BUILD STREAMLIT CHATBOT UI
# -----------------------------------------------
st.set_page_config(page_title="📚 AI Book Recommendation Chatbot", layout="wide")
st.title("📚 AI Book Recommendation Chatbot")
st.markdown("👋 Ask me for book recommendations based on genre, themes, or authors!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
user_query = st.chat_input("Ask me for book recommendations...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve book recommendations
    recommended_books = retrieve_similar_books(user_query)

    if recommended_books:
        # Generate response using Ollama
        response = generate_recommendation_response(user_query, recommended_books)
    else:
        response = "❌ Sorry, I couldn't find any matching books. Try a different genre or theme!"

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
