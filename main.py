import os
import requests
from dotenv import load_dotenv
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random

# Load API key
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Create data folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Fetch News Articles
def fetch_news():
    print("📰 Fetching news articles...")
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=100&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    return articles

# Save raw news
def save_news(articles):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"data/news_{today}.csv"
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)
    print(f"✅ News data saved in '{filename}'")

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

# Process and clean
def process_news():
    today = datetime.today().strftime('%Y-%m-%d')
    raw_file = f"data/news_{today}.csv"
    if not os.path.exists(raw_file):
        print(f"⚠️ No news file found for today: {raw_file}")
        return None
    df = pd.read_csv(raw_file)
    df = df[['title', 'description', 'content']].fillna('')
    df['full_text'] = df['title'] + " " + df['description'] + " " + df['content']
    df['clean_text'] = df['full_text'].apply(clean_text)
    cleaned_file = f"data/clean_news_{today}.csv"
    df.to_csv(cleaned_file, index=False)
    print(f"🧹 Cleaned news saved in '{cleaned_file}'")
    return df

# Recommend similar articles
def recommend_articles(df, article_index=0):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[article_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:6]]  # Top 5 similar excluding itself
    return df.iloc[sim_indices]

# Run the full pipeline
def run_pipeline():
    articles = fetch_news()
    if not articles:
        print("❌ No articles fetched.")
        return
    save_news(articles)
    df = process_news()
    if df is None or df.empty:
        print("❌ No data to recommend from.")
        return
    
    article_index = 0  # Or use random.randint(0, len(df)-1) for random article
    print(f"\n📌 Using article index {article_index} for recommendations.")
    print(f"\n🔎 Original Article:\n{df.iloc[article_index]['title']}")
    
    recommendations = recommend_articles(df, article_index)

    print("\n✨ Recommended Articles:\n")
    for idx, row in recommendations.iterrows():
        print(f"• {row['title']}")

# Execute
if __name__ == "__main__":
    run_pipeline()
