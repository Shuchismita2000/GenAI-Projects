import json
import joblib
from transformers import pipeline

# Load intents from JSON
def load_intents(json_path="../data/intents.json"):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data["intents"]

MODEL_PATH = r"D:\gen-ai-trial\GenAI-Projects\mental_health_bot\scripts\intent_classifier.sav"

# Check if model is already saved
try:
    intent_classifier = joblib.load(MODEL_PATH)  # Load from cache
    print("✅ Sentiment Model Loaded from Cache.")
except:
    print("⚠ Model not found! Initializing and saving a new intent classifier...")
    # Load Zero-Shot Classification Model
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    joblib.dump(intent_classifier, MODEL_PATH)  # Save for future use
    print("✅ Model Saved Successfully.")


# Get the list of intents from the dataset
intents = load_intents()
intent_labels = [intent["tag"] for intent in intents]

# Function to classify user input
def classify_intent(user_input):
    """
    Identifies the most likely intent based on user input.
    
    :param user_input: (str) The text input from the user.
    :return: (str) The classified intent tag.
    """
    result = intent_classifier(user_input, candidate_labels=intent_labels)
    return result["labels"][0]  # Return the most relevant intent

# Example Testing
if __name__ == "__main__":
    user_message = "I feel very lonely and sad."
    detected_intent = classify_intent(user_message)
    print(f"User Input: {user_message}")
    print(f"Detected Intent: {detected_intent}")
