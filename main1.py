import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv('NEWS_API_KEY')
QUERY = 'news'

# Fetch news articles for a given date
def fetch_news(date_str):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={QUERY}&"
        f"from={date_str}&to={date_str}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"pageSize=100&"
        f"apiKey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print(f"⚠️ Failed to fetch news for {date_str}: {response.status_code}")
        return []

# Save raw articles to CSV
def save_news(articles, date_str):
    if not articles:
        print(f"⚠️ No articles to save for {date_str}")
        return
    df = pd.DataFrame(articles)
    os.makedirs('data', exist_ok=True)
    df.to_csv(f"data/news_{date_str}.csv", index=False)
    print(f"✅ Saved {len(df)} articles to data/news_{date_str}.csv")

# Clean and keep only required fields
def process_news(date_str):
    try:
        df = pd.read_csv(f"data/news_{date_str}.csv")
        df_clean = df[['title', 'description', 'content', 'publishedAt', 'url', 'urlToImage']].fillna('')
        df_clean.to_csv(f"data/clean_news_{date_str}.csv", index=False)
        print(f"✅ Cleaned and saved articles for {date_str}")
    except Exception as e:
        print(f"⚠️ Error processing news for {date_str}: {e}")

# Delete old files (older than n days)
def cleanup_old_files(folder='data', days_to_keep=7):
    cutoff_date = datetime.today() - timedelta(days=days_to_keep)
    for filename in os.listdir(folder):
        if filename.startswith('news_') or filename.startswith('clean_news_'):
            try:
                date_str = filename.split('_')[1].split('.')[0]
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                if file_date < cutoff_date:
                    os.remove(os.path.join(folder, filename))
                    print(f"🗑️ Deleted old file: {filename}")
            except Exception as e:
                print(f"⚠️ Error deleting {filename}: {e}")

# Main function
def main():
    today = datetime.today()

    for i in range(7):  # Past 7 days
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n📅 Fetching news for {date_str}...")
        articles = fetch_news(date_str)
        if articles:
            save_news(articles, date_str)
            process_news(date_str)
    
    cleanup_old_files()

if __name__ == "__main__":
    main()
