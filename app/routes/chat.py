from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest
from app.services.agent import NewsAgentGraph

# Initialize router
chat_router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize the news agent (this will be imported in main.py)
news_agent = None

def set_news_agent(agent: NewsAgentGraph):
    """Set the news agent instance"""
    global news_agent
    news_agent = agent

@chat_router.post("/")
async def chat(request: ChatRequest):
    """Handle chat requests"""
    try:
        if news_agent is None:
            raise HTTPException(status_code=500, detail="News agent not initialized")
        
        response = news_agent.process_message(request.message, request.session_id)
        return {"response": response}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}