# 📚 **AI Book Recommendation Chatbot – Guidebook**

## **1. Introduction**

### **Overview**

This project is an AI-driven **book recommendation chatbot** that suggests books based on user queries. The chatbot leverages **natural language processing (NLP), embeddings, and retrieval-augmented generation (RAG)** to analyze book summaries, genres, and themes, providing relevant recommendations.

### **Objectives**

- Gather and structure book data.
- Generate embeddings for book descriptions to enable similarity search.
- Implement a recommendation system using **vector search** (Pinecone).
- Deploy an interactive chatbot for book recommendations.

### **Target Audience**

- **Readers** looking for book recommendations.
- **Developers & AI enthusiasts** interested in NLP-based recommendation systems.
- **Librarians & Bookstores** seeking AI-powered book search solutions.

---

## **2. Prerequisites**

### **Required Tools & Libraries**

- **Python 3.8+**
- **Pandas & NumPy** (for data processing)
- **NLTK / SpaCy** (for text cleaning & tokenization)
- **Hugging Face Sentence-Transformer** (for embeddings)
- **Pinecone** (vector database for similarity search)
- **Streamlit** (for chatbot UI)

## **3. Project Structure**

### 🎯 **Phase 1: Collect & Structure Book Data**

📌 ***Bronze Layer:* Source Book Data**  
Instead of fetching data from APIs, we **use a [Kaggle dataset](https://www.kaggle.com/datasets/abdallahwagih/books-dataset)** that contains book metadata such as:  
✅ **Title**  
✅ **Author**  
✅ **Genres & Themes**  
✅ **Description/Summary**  
✅ **User Ratings**

### 🎯 **Phase 2: Process & Embed Book Data**

To ensure high-quality book recommendations, we **process raw data** in two key transformation stages:

1. **Bronze → Silver Layer:** Cleans and standardizes the raw dataset.
2. **Silver → Gold Layer:** Enhances data for AI embedding and retrieval.

### **🔹 Bronze to Silver Data Cleaning Pipeline**

🎯 **Objective:** Clean and standardize raw book metadata for improved usability.

|Step|Task|Reasoning|
|---|---|---|
|**1**|Load the dataset|Import data into a structured format for processing.|
|**2**|Remove special characters & HTML tags|Ensures clean and readable text. Some sources include unwanted HTML tags.|
|**3**|Remove stopwords|Reduces noise in text fields, improving relevance in embeddings.|
|**4**|Convert text to lowercase|Standardizes text for consistency in comparisons.|
|**5**|Fill missing values|Prevents errors in downstream processes. Uses logical placeholders like `"Unknown Author"` instead of empty values.|
|**6**|Standardize numerical values (ratings, page count)|Ensures uniform representation for accurate analysis. For example, normalizing ratings to a **0-5 scale**.|
|**7**|Standardize date format (published year)|Maintains consistency across different sources. Converts **"Feb 5, 2001" → "2001"** for simplicity.|
|**8**|Clean thumbnail URLs|Fixes malformed URLs to ensure proper rendering in UI.|
|**9**|Handle erroneous data|Filters out corrupted or incomplete records (e.g., missing title/author).|

📌 **Silver Layer Output:** A structured, clean dataset **ready for advanced processing**.

### **🔹 Silver to Gold Data Enhancement Pipeline**

🎯 **Objective:** Prepare data for **embedding generation** and AI-powered recommendations.

|Step|Task|Reasoning|
|---|---|---|
|**1**|Fix malformed URLs|Ensures images and external links work correctly in the chatbot UI.|
|**2**|Normalize genre categories|Standardizes genre labels across different datasets (e.g., "Sci-fi" → "Science Fiction").|
|**3**|Remove short/incomplete descriptions|Books with very short descriptions (< 30 words) may lack context for embeddings.|
|**4**|Truncate overly long text|Embeddings have length limitations; excessive text can cause inefficiencies.|
|**5**|Create a unified text field|Combines **title, author, genres, and description** into a single **semantic field** for embeddings.|

📌 **Gold Layer Output:** A refined dataset **optimized for AI-based recommendations**.


### **🔹Generate Embeddings with LangChain**

We will use **LangChain's Hugging Face integration** to generate numerical representations (**embeddings**) of book descriptions.

📌 **Why `all-MiniLM-L6-v2`?**

- Lightweight and **efficient for semantic search**.
- Captures **contextual meaning** of text.
- **512-dimension embeddings** for **fast & accurate similarity search**.
### **🔹Store Embeddings in Pinecone**

Next, we **store these embeddings** in **Pinecone**, allowing **fast similarity-based retrieval**.


### 🎯 **Phase 3: Define the RAG Pipeline**

The RAG system will:

1. **Connect to Pinecone** to retrieve stored book embeddings.
2. **Convert user queries** into embeddings using `all-MiniLM-L6-v2`.
3. **Perform a vector similarity search** in Pinecone to find relevant books.
4. **Retrieve metadata** for the top recommended books.
5. **Use Ollama - Mistral** to generate a **conversational response** incorporating the retrieved book information.



### 🎯 **Phase 4: Create a Chat Interface**

- Use **Streamlit** for an interactive user interface.
- Display **dynamic recommendations** based on user queries.


## **🔹 Key Differences Between the Two Chatbots**

|Feature|**6.book_chatbot.py**|**6.2.book_chatbot.py**|
|---|---|---|
|**Query Classification**|No query classification|**Classifies user queries** (recommendation, book info, summary, genre discussion, general chat)|
|**Recommendation Handling**|Retrieves books and directly sends them to Ollama|**Formats recommendations before passing to Ollama** for response generation|
|**Chat History Usage**|Does not include chat history in AI-generated responses|**Includes previous messages in AI responses** to enable conversational flow|
|**Book Recommendation Logic**|Only retrieves books and asks Ollama to generate a response|**Formats recommendations into structured text** before passing to Ollama|
|**Error Handling**|Returns a static message if no books are found|Provides an alternative response with a **fallback chat** instead of failing|
|**Interaction Style**|**Straightforward Q&A chatbot**|**Conversational chatbot** that encourages discussions|

---

## **🔹 Breakdown of Each Chatbot**

### **1️⃣ 6.book_chatbot.py - Simple Book Recommendation Chatbot**

📌 **What it does:**

- **Accepts a query** and retrieves book recommendations from Pinecone.
- **Formats book recommendations** and sends them directly to Ollama.
- **Displays recommendations** without further conversation handling.

📌 **Strengths:** 

✅ **Faster response time** since it retrieves books and sends them directly to Ollama.  
✅ **Simple and efficient** for users who just want book suggestions.  
✅ **Less processing overhead** (no complex chat history management).

📌 **Limitations:** 

❌ **No query classification** – does not differentiate between types of book-related queries.  
❌ **No conversational continuity** – every response is independent, making it less interactive.  
❌ **No fallback responses** – if no books are found, it just states that.

#### **Reasoning Behind Its Simplicity**

This chatbot is great if users **only need quick book recommendations**. It does not handle **deeper conversations** about book summaries, genres, or authors.

---

### **2️⃣ 6.2.book_chatbot.py - Conversational AI Book Chatbot**

📌 **What it does:**

- **Classifies the user query** (e.g., recommendation, book info, genre, summary, or general chat).
- **Retrieves relevant books** based on query intent.
- **Enhances conversational experience** by integrating previous chat history into responses.
- **Generates more dynamic AI responses** by passing structured book recommendations into Ollama.

📌 **Strengths:**

✅ **Engages in discussions** instead of just listing books.  
✅ **Understands different query types** (not just book recommendations).  
✅ **Uses chat history** to improve **conversation flow**.  
✅ **Encourages follow-up questions** rather than just returning static book lists.  
✅ **Handles alternative responses** if no relevant books are found.

📌 **Limitations:** 

❌ **Slightly slower response time** due to query classification and chat history integration.  
❌ **More computationally expensive** (as it makes dynamic queries to Ollama).

#### **Reasoning Behind the Advanced Interaction**
![Demo](https://drive.google.com/file/d/1x2wLQAsEhFDRkaUewwkKEc8zUaaDeAI2/view?usp=sharing)
This chatbot is ideal for users who want **an AI book assistant** that goes beyond recommendations, **engaging in meaningful book discussions**.
