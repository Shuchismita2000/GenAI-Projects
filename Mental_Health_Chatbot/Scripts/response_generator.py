import ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama
from sentiment_analysis import analyze_sentiment

# Initialize memory for LangChain
memory = ConversationBufferMemory()

# Load Mistral 7B model via LangChain
llm = ChatOllama(model="mistral", temperature=0.7)
conversation = ConversationChain(llm=llm, memory=memory)

def generate_response(user_input):
    """
    Generates an AI response using a combination of sentiment analysis, memory, and LLM.
    
    :param user_input: (str) The user's message.
    :return: (str) Generated chatbot response.
    """
    # Analyze sentiment
    sentiment, confidence = analyze_sentiment(user_input)

    # Respond based on sentiment
    if sentiment == "NEGATIVE":
        return "I'm really sorry you're feeling this way. You're not alone, and I'm here to listen. Do you want to talk about what's on your mind?"
    

    # Use LangChain memory & Mistral for neutral/general responses
    bot_response = conversation.run(user_input)

    return bot_response

# Example test
if __name__ == "__main__":
    user_input = "I'm feeling really down today."
    bot_response = generate_response(user_input)
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")

    user_input = "I just had a great day at work!"
    bot_response = generate_response(user_input)
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")

    user_input = "Can you tell me about mindfulness?"
    bot_response = generate_response(user_input)
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")
