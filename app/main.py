from fastapi import FastAPI
from app.models.base import create_tables
from app.services.agent import NewsAgentGraph
from app.services.news_scraper import NewsScraper
from app.routes.chat import chat_router, set_news_agent
from app.routes.news import news_router, set_scraper

# Create FastAPI app
app = FastAPI(
    title="Smart News Chat Bot",
    description="An intelligent news chat bot with search, summarization, translation, and PDF generation capabilities",
    version="1.0.0"
)

# Initialize services
news_agent = NewsAgentGraph()
scraper = NewsScraper()

# Set service instances in routers
set_news_agent(news_agent)
set_scraper(scraper)

# Include routers
app.include_router(chat_router)
app.include_router(news_router)

@app.get("/")
def home():
    """Root endpoint"""
    return {
        "message": "🤖 Smart News Chat Bot is ready! Use /news/scrape to get started.",
        "endpoints": {
            "chat": "/chat/",
            "scrape_news": "/news/scrape/",
            "health": "/news/health/",
            "docs": "/docs"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("🚀 Starting Smart News Chat Bot...")
    
    # Create database tables
    create_tables()
    print("✅ Database tables created/verified")
    
    # Auto-scrape news on startup
    print("🔄 Auto-scraping news on startup...")
    result = scraper.scrape_and_store()
    print(result)
    
    print("🎉 Smart News Chat Bot is ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    print("🛑 Shutting down Smart News Chat Bot...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)