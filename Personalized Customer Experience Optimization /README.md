## **Intelligent Product & Order Support Assistant**

### **Domain**:

**E-commerce (Beauty & Personal Care Products)**

### **Problem Statement**:

> **"Develop an intelligent customer support agent for an e-commerce platform (e.g., Mamaearth) that answers product, order, and promotion-related questions using a fine-tuned LLM, with LangGraph managing conversation memory, product lookup, and feedback-driven learning."**

---

## Use Case Overview:

Customers often ask about:

- Product features, ingredients, and suitability (e.g., “Is this safe for sensitive skin?”)
    
- Order history or refund status
    
- Current promotions and sales
    
- Inventory availability
    
- Subscription/cancellation
    

Instead of relying on human agents, this project simulates a **smart, AI-powered assistant** using real-looking data and structure.

---

## Your Data Files as Inputs:

|File|Use|
|---|---|
|`customer_purchase_data.csv`|Context for orders, refunds, etc.|
|`customer_demographic_data_with_real_names.csv`|Personalization (e.g., skin type, gender-based queries)|
|`mamaearth_products.csv`|Core for product Q&A|
|`mamaearth_inventory.csv`|To answer "is this in stock?"|
|`sales_trends_and_promotions.csv`|Surface discounts, upsells|
|`customer_purchase_data_large.csv`|Use for fine-tuning more robustly|

---

## **LangGraph Use in Flow**:



`Customer Input →    [Node 1] Intent Classifier (Product Q / Order Q / Promo Q) →    [Node 2] Memory (has this user asked before?) →    [Node 3] Product Lookup or Order Query →    [Node 4] LLM Generates Answer →    [Node 5] Ask for Feedback (Helpful or Not) →    [Node 6] Store poor responses for fine-tuning`

---

##  **Fine-tuned LLM Task**:

- Fine-tune a model (like Mistral or LLaMA) using synthetic customer support dialogues:
    
    - **Product FAQ conversations**
        
    - **Order status Q&A**
        
    - **Refund/cancellation dialogue**
        
- Personalize answers using demographic context (e.g., skin type, age, purchase behavior).
    

---

## Sample Questions the Agent Should Handle:

- “Is this face wash suitable for oily skin?”
    
- “When will my shampoo order arrive?”
    
- “I applied a coupon but didn’t get a discount.”
    
- “Suggest something for dry scalp.”
    
- “You didn’t help me. Try again.”
    

---

## Feedback Loop:

- Collect feedback on answers ("Helpful" / "Not Helpful").
    
- If not helpful:
    
    - Retry with revised prompt
        
    - Route to knowledge base or escalate
        
    - Log it for re-finetuning (RLHF-style)
        

---
## Project Architecture (Components + Flow)
```
                     ┌────────────────────────┐
                     │   Customer Frontend    │
                     │  (Streamlit/Gradio UI) │
                     └────────────┬───────────┘
                                  │
                   Customer Input│
                                  ▼
                     ┌────────────────────────┐
                     │   LangGraph Flow       │
                     └────────────┬───────────┘
                                  ▼
                ┌────────────────────────────────┐
                │     Node 1: Intent Classifier   │ ← Classify into:
                └────────────┬───────────────────┘     [Product / Order / Promo]
                             ▼
         ┌─────────────────────────────┐
         │ Node 2: Memory & Context    │ ← LangGraph memory
         └────────────┬────────────────┘
                      ▼
          ┌─────────────────────────────┐
          │ Product Lookup / Order Data │ ← Pulls from CSV or DB
          └────────────┬────────────────┘
                       ▼
            ┌──────────────────────────────┐
            │ Node 4: LLM Response Generator│ ← Fine-tuned LLM
            └────────────┬─────────────────┘
                         ▼
            ┌──────────────────────────────┐
            │ Node 5: Feedback Collector   │ ← "Helpful" or not
            └────────────┬─────────────────┘
                         ▼
            ┌──────────────────────────────┐
            │ Node 6: Log Poor Responses   │ → Used for retraining
            └──────────────────────────────┘


```