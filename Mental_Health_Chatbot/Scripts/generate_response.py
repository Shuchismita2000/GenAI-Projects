import ollama
import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama
from sentence_transformers import SentenceTransformer
from sentiment_analysis import analyze_sentiment
from intent_classifier import classify_intent

# 🔹 Initialize Pinecone
PINECONE_API_KEY = "pcsk_6WjJ7w_Mdge2WY3AVCqN5RafTo3byPqv7ECMdRSr1SZ4oS6quNbkDb4Lc7tnE2esM9DKQe"
INDEX_NAME = "mental-health-bot"

pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index = pinecone_client.Index(INDEX_NAME)

# 🔹 Load Sentence Transformer Model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Initialize LangChain memory
memory = ConversationBufferMemory()

# 🔹 Load Mistral 7B model via LangChain
llm = ChatOllama(model="mistral", temperature=0.7)
conversation = ConversationChain(llm=llm, memory=memory)

# 🔹 Function to Retrieve FAQ from Pinecone
def retrieve_faq(context):
    """
    Searches Pinecone for the most relevant FAQ answer.
    
    :param question: (str) The user's input question.
    :return: (str) The best-matching FAQ answer if found, else None.
    """
    query_vector = embed_model.encode(context).tolist()  # Convert query to vector
    result = index.query(vector=query_vector, top_k=1, include_metadata=True)

    if result["matches"]:
        return result["matches"][0]["metadata"]["Response"]
    
    return None  # No match found

# 🔹 Generate AI Response
def generate_response(user_input):
    """
    Hybrid chatbot logic:
    - Uses sentiment analysis for empathy.
    - Uses intent classification to prioritize response types.
    - Searches Pinecone for FAQ-based answers.
    - Uses Mistral + LangChain memory for open conversation.
    """
    # 1️⃣ Detect Intent
    intent = classify_intent(user_input)

    # 2️⃣ Sentiment Analysis
    sentiment, confidence = analyze_sentiment(user_input)
    
    if sentiment == "NEGATIVE":
        return "I'm really sorry you're feeling this way. You're not alone, and I'm here to listen. Do you want to talk about what's on your mind?"
    
    elif sentiment == "POSITIVE":
        return "Tell me more about your day. 😊"

    # 3️⃣ Check Pinecone FAQ Retrieval (for knowledge-based intents)
    if intent in ["faq", "information", "mental_health_facts"]:
        faq_response = retrieve_faq(user_input)
        if faq_response:
            return faq_response  # Return FAQ response if found

    # 4️⃣ Use LangChain memory & Mistral for general conversation
    bot_response = conversation.run(user_input)

    return bot_response

# 🔹 Example Test
if __name__ == "__main__":
    test_inputs = [
        "I'm feeling really down today.",
        "I just had a great day at work!",
        "What are the symptoms of depression?",
        "Can you tell me about mindfulness?"
    ]

    for user_input in test_inputs:
        bot_response = generate_response(user_input)
        print(f"User Input: {user_input}")
        print(f"Bot Response: {bot_response}\n")
