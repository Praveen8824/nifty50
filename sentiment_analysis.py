"""
Sentiment Analysis Module for Stock Market News
"""
import requests
import os
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from config import NEWS_API_KEY


class SentimentAnalyzer:
    """Class for analyzing sentiment from news articles"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.api_key = NEWS_API_KEY
    
    def get_news(self, query, language='en', sort_by='relevancy', page_size=20):
        """
        Fetch news articles from News API
        
        Args:
            query: Search query
            language: Language code (default: 'en')
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of articles to fetch
        
        Returns:
            List of news articles
        """
        if not self.api_key:
            return []
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'apiKey': self.api_key,
                'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                print(f"News API Error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def get_stock_sentiment(self, stock_name, stock_symbol):
        """
        Get sentiment for a specific stock
        
        Args:
            stock_name: Name of the stock (e.g., 'Reliance')
            stock_symbol: Stock symbol (e.g., 'RELIANCE')
        
        Returns:
            Dictionary with sentiment analysis results and articles
        """
        # Search queries for relevant news
        queries = [
            f"{stock_name} stock",
            f"{stock_symbol} earnings",
            f"{stock_name} Q1 Q2 Q3 Q4 results",
            f"{stock_name} financial results",
            f"{stock_name} quarterly results"
        ]
        
        all_articles = []
        for query in queries:
            articles = self.get_news(query, page_size=10)
            all_articles.extend(articles)
        
        # Remove duplicates
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Analyze sentiment with articles
        article_sentiments = []
        for article in unique_articles[:30]:  # Limit to 30 articles
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            sentiment = self.analyze_sentiment(content)
            article_sentiments.append({
                'title': title,
                'description': description,
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'sentiment': sentiment['compound'],
                'sentiment_label': 'bullish' if sentiment['compound'] >= 0.05 else 'bearish' if sentiment['compound'] <= -0.05 else 'neutral'
            })
        
        if not article_sentiments:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'articles_count': 0,
                'articles': []
            }
        
        sentiments = [a['sentiment'] for a in article_sentiments]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Classify sentiment
        if avg_sentiment >= 0.05:
            sentiment_label = 'bullish'
        elif avg_sentiment <= -0.05:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment': sentiment_label,
            'score': avg_sentiment,
            'articles_count': len(unique_articles),
            'articles': article_sentiments[:20]  # Return top 20 articles
        }
    
    def get_market_sentiment(self, market_type='indian'):
        """
        Get overall market sentiment
        
        Args:
            market_type: 'indian', 'global', or 'both'
        
        Returns:
            Dictionary with market sentiment
        """
        queries = []
        
        if market_type == 'indian' or market_type == 'both':
            queries.extend([
                'Indian stock market',
                'Nifty 50',
                'BSE Sensex',
                'Indian economy'
            ])
        
        if market_type == 'global' or market_type == 'both':
            queries.extend([
                'global stock market',
                'world economy',
                'stock market trends'
            ])
        
        all_articles = []
        for query in queries:
            articles = self.get_news(query, page_size=15)
            all_articles.extend(articles)
        
        # Remove duplicates
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Analyze sentiment
        sentiments = []
        for article in unique_articles[:30]:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            sentiment = self.analyze_sentiment(content)
            sentiments.append(sentiment['compound'])
        
        if not sentiments:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'articles_count': 0
            }
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment >= 0.05:
            sentiment_label = 'bullish'
        elif avg_sentiment <= -0.05:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment': sentiment_label,
            'score': avg_sentiment,
            'articles_count': len(unique_articles)
        }

