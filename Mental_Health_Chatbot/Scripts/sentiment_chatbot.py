from models.sentiment_analysis import analyze_sentiment  
def generate_chatbot_response(user_input):     
    """     Generates a chatbot response based on sentiment analysis.          
    :param user_input: (str) The message from the user.     
    :return: (str) Chatbot's response.     
    """     
    sentiment, confidence = analyze_sentiment(user_input)      
    if sentiment == "NEGATIVE":         
        return "I'm really sorry you're feeling this way. Do you want to talk about it?"     
    elif sentiment == "POSITIVE":         
        return "That's great to hear! Tell me more about what's making you happy today."     
    else:         
        return "I'm here to listen. What's on your mind?"  
    
    
# Example test 
if __name__ == "__main__":     
    user_input = "I'm feeling really down today."     
    bot_response = generate_chatbot_response(user_input)
    print(f"Bot Response: {bot_response}")