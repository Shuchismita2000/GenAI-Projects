from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama

# Initialize Memory
memory = ConversationBufferMemory()

# Load Ollama Chat Model (Mistral 7B)
llm = ChatOllama(model="mistral", temperature=0.7)

# Conversation Chain (Maintains Chat History)
conversation = ConversationChain(llm=llm, memory=memory)

def generate_response_with_memory(user_input):
    """
    Generates a response while maintaining chat memory using Ollama.
    
    :param user_input: (str) The user's message.
    :return: (str) Generated chatbot response.
    """
    return conversation.run(user_input)

# Example test
if __name__ == "__main__":
    user_input = "I feel really sad today."
    bot_response = generate_response_with_memory(user_input)
    print(f"Bot Response: {bot_response}")
