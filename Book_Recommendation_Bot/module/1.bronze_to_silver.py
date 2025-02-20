"""
📌 Bronze to Silver Data Cleaning Pipeline
 Step 1: Load the dataset
Step 2: Remove special characters & HTML tags
Step 3: Remove stopwords
Step 4: Convert text to lowercase
Step 5: Fill missing values with logical placeholders
Step 6: Standardize numerical values (ratings, page count)
Step 7: Standardize date format (published year)
Step 8: Clean thumbnail URLs
Step 9: Handle erroneous data
"""

import pandas as pd
import re

# ============================
# Step 1: Load the dataset
# ============================
def load_data(file_path):
    """Load the CSV dataset into a Pandas DataFrame."""
    return pd.read_csv(file_path)

# ============================
# Step 2: Clean Text - Remove special characters & HTML tags
# ============================
def clean_text(text):
    """Remove special characters and HTML tags from text columns."""
    if isinstance(text, str):
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)  # Keep only meaningful characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# ============================
# Step 3: Remove Stopwords
# ============================
manual_stopwords = {
    "the", "and", "a", "in", "of", "to", "is", "that", "it", "on", "for", "with",
    "as", "was", "at", "by", "an", "be", "this", "which", "or", "from", "but", "are",
    "not", "were", "have", "has", "had", "will", "would", "can", "could", "should",
    "you", "your", "we", "our", "they", "their", "his", "her", "its", "i", "me", "my",
    "he", "she", "him", "them", "us", "about", "so", "just", "if", "because", "what",
    "when", "how", "where", "who", "whom", "why", "then", "than", "now", "some",
    "any", "each", "all", "most", "many", "few", "more", "other", "such"
}

def remove_stopwords(text):
    """Remove common stopwords from text."""
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in manual_stopwords]
        return " ".join(filtered_words)
    return text

# ============================
# Step 4: Convert Text to Lowercase
# ============================
def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower() if isinstance(text, str) else text

# ============================
# Step 5: Fill Missing Values
# ============================
def fill_missing_values(df):
    """Fill missing values with logical placeholders."""
    fill_values = {
        'authors': "Unknown author",
        'description': "No description available",
        'categories': "Uncategorized",
        'subtitle': "No subtitle",
        'thumbnail': "No image available",
        'published_year': -1,  # Placeholder for missing year
        'average_rating': 0.0,  # Default rating if missing
        'num_pages': 0,  # Default page count if missing
        'ratings_count': 0  # Default ratings count if missing
    }
    return df.fillna(value=fill_values)

# ============================
# Step 6: Standardize Numerical Data
# ============================
def standardize_numerical_data(df):
    """Ensure ratings, page counts, and rating counts are within reasonable ranges."""
    df['average_rating'] = df['average_rating'].clip(lower=0, upper=5)
    df['num_pages'] = df['num_pages'].clip(lower=1, upper=5000)
    df['ratings_count'] = df['ratings_count'].clip(lower=0, upper=10_000_000)
    return df

# ============================
# Step 7: Standardize Date Format
# ============================
def standardize_dates(df):
    """Ensure the published year is in integer format and handle missing years."""
    df['published_year'] = df['published_year'].fillna(-1).astype(int)
    df['published_year'] = df['published_year'].replace(-1, "Unknown Year")
    return df

# ============================
# Step 8: Clean Thumbnail URLs
# ============================
def clean_thumbnail_urls(df):
    """Ensure thumbnail URLs are properly formatted."""
    df['thumbnail'] = df['thumbnail'].apply(lambda x: x if isinstance(x, str) and x.startswith("http") else "No image available")
    return df

# ============================
# Step 9: Handle Erroneous Data
# ============================
def handle_erroneous_data(df):
    """Ensure that data doesn't contain anomalies such as negative pages or ratings."""
    df.loc[df['num_pages'] < 1, 'num_pages'] = 1  # Ensure at least 1 page
    df.loc[df['average_rating'] < 0, 'average_rating'] = 0  # No negative ratings
    return df

# ============================
# Pipeline Execution
# ============================
def bronze_to_silver_pipeline(file_path):
    """Runs the full data cleaning pipeline from Bronze to Silver."""
    # Load data
    df = load_data(file_path)

    # Identify text columns
    text_columns = df.select_dtypes(include=['object']).columns

    # Apply text cleaning
    df['description'] = df['description'].apply(clean_text)
    #df[text_columns] = df[text_columns].applymap(remove_stopwords)
    df[text_columns] = df[text_columns].applymap(to_lowercase)

    # Fill missing values
    df = fill_missing_values(df)

    # Standardize numerical data
    df = standardize_numerical_data(df)

    # Standardize date format
    df = standardize_dates(df)

    # Clean thumbnail URLs
    df = clean_thumbnail_urls(df)

    # Handle erroneous data
    df = handle_erroneous_data(df)

    return df

# ============================
# Run the Pipeline (Example)
# ============================
if __name__ == "__main__":
    input_file = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\bronze.csv"  # Replace with actual file path
    df_cleaned = bronze_to_silver_pipeline(input_file)

    # Save the cleaned data
    output_file = r"D:\gen-ai-trial\GenAI-Projects\book_rec_bot\data\silver.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"✔ Data cleaning completed. Cleaned dataset saved as '{output_file}'")
