import ollama

def generate_response(user_input):
    """
    Generates an AI response using Ollama with Mistral.

    :param user_input: (str) The user's message.
    :return: (str) Generated chatbot response.
    """
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": user_input}])
    return response["message"]["content"]

# Example test
if __name__ == "__main__":
    user_input = "I feel really lonely today."
    bot_response = generate_response(user_input)
    print(f"User Input: {user_input}")
    print(f"Bot Response: {bot_response}")
