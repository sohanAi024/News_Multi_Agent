import os
import requests
from sentence_transformers import SentenceTransformer
from app.models.base import SessionLocal
from app.models.news_document import NewsDocument
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv
from config import Config

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

llm = ChatMistralAI(api_key=Config.MISTRAL_API_KEY)

def classify_category_with_mistral(text: str) -> str:
    prompt = f"""
    Analyze this news article and provide ONE short category (1-2 words)
    that best represents the topic. Examples: Technology, Politics, Health, Sports, Business, Entertainment, Science, World, Education, Finance.

    News Content:
    {text}

    Respond with only the category name, nothing else.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


class NewsScraper:
    def __init__(self):
        self.session = SessionLocal()

    def scrape_and_store(self):
        """
        Scrape news from NewsAPI and store in DB with category and embedding.
        """
        try:
            url = f'https://newsapi.org/v2/top-headlines?language=en&pageSize=50&apiKey={Config.NEWS_API_KEY}'
            response = requests.get(url).json()
            new_count = 0

            for article in response.get('articles', []):
                title = article.get('title', '')
                desc = article.get('description', '')
                link = article.get('url', '')

                if not title or not link:
                    continue

                # Skip if already exists
                if self.session.query(NewsDocument).filter_by(url=link).first():
                    continue

                # Combine title and description for content
                content = f"{title}. {desc}" if desc else title

                # ✅ Generate embedding
                embedding = embedding_model.encode(content).tolist()

                # ✅ Detect category dynamically using Mistral
                category = classify_category_with_mistral(content)

                # ✅ Save to DB
                self.session.add(NewsDocument(
                    title=title,
                    content=content,
                    url=link,
                    category=category,
                    embedding=embedding
                ))
                new_count += 1

            self.session.commit()
            return f"✅ {new_count} new articles stored successfully"

        except Exception as e:
            self.session.rollback()
            return f"❌ Error during scraping: {str(e)}"

        finally:
            self.session.close()
