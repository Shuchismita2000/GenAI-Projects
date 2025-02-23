from transformers import pipeline
import joblib

MODEL_PATH = r"D:\gen-ai-trial\GenAI-Projects\mental_health_bot\scripts\sentiment_model.sav"

# Check if model is already saved
try:
    sentiment_analyzer = joblib.load(MODEL_PATH)  # Load from cache
    print("✅ Sentiment Model Loaded from Cache.")
except:
    print("⚠ Model not found! Initializing and saving a new sentiment analyzer...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    joblib.dump(sentiment_analyzer, MODEL_PATH)  # Save for future use
    print("✅ Model Saved Successfully.")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text.
    
    :param text: (str) User's message.
    :return: (tuple) (Sentiment Label, Confidence Score)
    """
    result = sentiment_analyzer(text)[0]  # Extract first result
    return result["label"], result["score"]

# Example test
if __name__ == "__main__":
    test_message = "I feel really anxious and depressed today."
    sentiment, confidence = analyze_sentiment(test_message)
    print(f"User Input: {test_message}")
    print(f"Detected Sentiment: {sentiment} (Confidence: {confidence:.2f})")
