from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Initialize Gemini Client
client = genai.Client(api_key="AIzaSyAT0XBdVWzV16zDcBmu33o6VBnsfMrFeio")
model_id = "gemini-2.0-flash"

# Define Google Grounding Search Tool
google_search_tool = Tool(google_search=GoogleSearch())

# FinBERT setup
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

# Search for financial news using grounding
def fetch_articles(query="Tesla stock", num=3):
    response = client.models.generate_content(
        model=model_id,
        contents=(
            f"Give me {num} short and distinct financial news summaries about {query}. "
            "Format them clearly like:\n\n1. <summary 1>\n2. <summary 2>\n3. <summary 3>"
        ),
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )

    raw_text = "\n".join([
        part.text for part in response.candidates[0].content.parts
        if hasattr(part, 'text')
    ])

    # Split based on numbered format
    import re
    articles = re.split(r"\n?\d+\.\s+", raw_text)
    articles = [a.strip() for a in articles if len(a.strip()) > 30]

    return articles[:num]

# FinBERT sentiment inference
def get_sentiment_embeddings(texts):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    sentiment_vectors = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze().numpy()
        sentiment_vector = softmax(logits)  # [negative, neutral, positive]
        sentiment_vectors.append(sentiment_vector)
    return sentiment_vectors

# Run the pipeline
if __name__ == "__main__":
    articles = fetch_articles("Tesla stock", num=7)
    
    print("🔍 Fetched Articles:")
    for i, article in enumerate(articles):
        print(f"\n--- Article {i+1} ---\n{article}")

    sentiments = get_sentiment_embeddings(articles)

    print("\n📊 Sentiment Scores:")
    for i, vec in enumerate(sentiments):
        print(f"Article {i+1} Sentiment (neg/neu/pos): {vec}")
