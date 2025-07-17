from fastapi import APIRouter
from app.services.news_scraper import NewsScraper

# Initialize router
news_router = APIRouter(prefix="/news", tags=["news"])

# Initialize scraper (this will be imported in main.py)
scraper = None

def set_scraper(news_scraper: NewsScraper):
    """Set the news scraper instance"""
    global scraper
    scraper = news_scraper

@news_router.post("/scrape/")
async def scrape_news():
    """Scrape and store news articles"""
    if scraper is None:
        return {"message": "❌ News scraper not initialized"}
    
    result = scraper.scrape_and_store()
    return {"message": result}

@news_router.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "News service is running"}