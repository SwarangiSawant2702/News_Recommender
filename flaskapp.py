from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import string
import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

app = Flask(__name__)
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Clean text for embedding
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join([word for word in text.split() if word not in stopwords])

# Fetch today's news
def fetch_today_news():
    today = datetime.today().strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, f"news_{today}.csv")
    
    if os.path.exists(file_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d')
        if file_mod_time == today:  # Check if the file is for today
            return file_path

    print("📥 Fetching today's news...")
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"language=en&country=us&pageSize=100&apiKey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch news: {response.status_code}")

    articles = response.json().get('articles', [])
    df = pd.DataFrame(articles)
    df['clean_text'] = df['title'].fillna('').apply(clean_text)
    df.to_csv(file_path, index=False)
    return file_path

# Load past 7 days of news
def load_past_news():
    all_data = []
    today = datetime.today()
    for i in range(1, 8):
        day = today - timedelta(days=i)
        path = os.path.join(DATA_DIR, f"news_{day.strftime('%Y-%m-%d')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['fetched_date'] = day.strftime('%Y-%m-%d')
            all_data.append(df[['title', 'description', 'url', 'fetched_date']])
    return pd.concat(all_data, ignore_index=True).fillna("") if all_data else pd.DataFrame()

# Recommend from any set of news data
def recommend_articles(user_input, df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['clean_text'] = df['title'].fillna('').apply(clean_text)

    article_embeddings = model.encode(df['clean_text'].tolist(), convert_to_tensor=True)
    input_embedding = model.encode([clean_text(user_input)], convert_to_tensor=True)

    sim_scores = cosine_similarity(input_embedding, article_embeddings)[0]
    top_indices = sim_scores.argsort()[-5:][::-1]
    recommended = df.iloc[top_indices]

    return [
        {
            "title": row.get('title', 'No Title'),
            "description": row.get('description', 'No Description'),
            "url": row.get('url', '#')
        }
        for _, row in recommended.iterrows()
    ]

# Home route - This is the homepage that renders `index.html`
@app.route('/')
def index():
    return render_template('index.html')

# POST: Recommend articles (from either today's news or past news)
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    date_range = request.form.get('date_range', 'today')  # 'today' or 'past'
    
    if date_range == 'today':
        file = fetch_today_news()
        df = pd.read_csv(file)
    else:
        df = load_past_news()

    recommendations = recommend_articles(user_input, df)
    return jsonify({"recommendations": recommendations})

# GET: Past 7 days
@app.route('/past-news')
def past_news():
    df = load_past_news()

    # Get filters
    filter_date = request.args.get('date')
    category = request.args.get('category', '').lower()

    # Apply date filter
    if filter_date:
        df = df[df['fetched_date'] == filter_date]

    # Apply category filter
    if category:
        df = df[
            df['title'].str.lower().str.contains(category, na=False) |
            df['description'].str.lower().str.contains(category, na=False)
        ]

    return jsonify({"articles": df.to_dict(orient='records')})

@app.route('/recommend-past', methods=['POST'])
def recommend_past():
    user_input = request.form['user_input']
    df = load_past_news()
    if df.empty:
        return jsonify({"recommendations": []})
    recommendations = recommend_articles(user_input, df)
    return jsonify({"recommendations": recommendations})

# GET: Today's news (raw)
@app.route('/today-news')
def today_news():
    file = fetch_today_news()
    df = pd.read_csv(file)
    return jsonify({"articles": df[['title', 'description', 'url']].fillna('').to_dict(orient='records')})

# GET: Past news HTML page
@app.route('/past-news.html')
def past_news_html():
    return render_template('past-news.html')


if __name__ == '__main__':
    app.run(debug=True)
