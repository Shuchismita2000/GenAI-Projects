# ==============================================
# 📚 AI BOOK RECOMMENDATION CHATBOT (CONVERSATIONAL)
# ==============================================

# Import necessary libraries
import os
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
    """Converts a user query into an embedding."""
    return embedding_model.encode(query).tolist()

# -----------------------------------------------
# 🔹 STEP 3: CLASSIFY USER QUERY TYPE
# -----------------------------------------------
def classify_query(user_query):
    """
    Classifies user intent based on the query type.
    """
    query_lower = user_query.lower()

    if any(keyword in query_lower for keyword in ["recommend", "suggest", "find"]):
        return "recommendation"
    elif any(keyword in query_lower for keyword in ["who wrote", "author of", "when was"]):
        return "book_info"
    elif any(keyword in query_lower for keyword in ["summary", "explain", "what is"]):
        return "summary"
    elif any(keyword in query_lower for keyword in ["genre", "type of book", "similar to"]):
        return "genre_info"
    else:
        return "general_chat"

# -----------------------------------------------
# 🔹 STEP 4: SEARCH PINECONE FOR SIMILAR BOOKS
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
# 🔹 STEP 5: GENERATE AI RESPONSE USING OLLAMA
# -----------------------------------------------
def generate_response(user_query, recommended_books, chat_history):
    """
    Uses Ollama to generate a conversational response based on user queries.
    """

    # Format book recommendations (if available)
    if recommended_books:
        book_list = "\n".join([
            f"📖 {book['title']} by {book['author']} (Genre: {book['genre']})"
            for book in recommended_books
        ])
        book_info = f"Here are some recommended books:\n{book_list}"
    else:
        book_info = "I couldn't find exact matches, but I can still chat about books!"

    # Define the enhanced conversational prompt
    prompt = f"""
    You are an AI book expert that not only recommends books but also engages in conversations about books, authors, genres, and literary discussions.

    **User's Query:** "{user_query}"
    
    **Book Recommendations (if applicable):** {book_info}

    **Chat History:**
    {chat_history}

    Respond in a friendly, engaging way, encouraging discussion and follow-up questions.
    """
    
    # Get AI-generated response from Ollama
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

# -----------------------------------------------
# 🔹 STEP 6: BUILD STREAMLIT CHATBOT UI
# -----------------------------------------------
st.set_page_config(page_title="📚 AI Book Recommendation Chatbot", layout="wide")
st.title("📚 AI Book Recommendation Chatbot")
st.markdown("👋 Ask me for book recommendations, summaries, author info, or general book discussions!")

# Initialize chat history if not already stored
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
user_query = st.chat_input("Ask me about books, authors, genres, or get recommendations!")

if user_query:
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Classify the user query type
    query_type = classify_query(user_query)

    # Process the query based on its type
    if query_type == "recommendation":
        recommended_books = retrieve_similar_books(user_query)
        response = generate_response(user_query, recommended_books, chat_history=st.session_state.messages)

    elif query_type == "book_info":
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Provide details about the book related to: {user_query}"}])["message"]["content"]

    elif query_type == "summary":
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Summarize the book: {user_query}"}])["message"]["content"]

    elif query_type == "genre_info":
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Tell me about the genre or similar books related to: {user_query}"}])["message"]["content"]

    else:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Engage in a book-related conversation. User asked: {user_query}"}])["message"]["content"]

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


