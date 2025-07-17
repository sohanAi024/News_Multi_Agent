from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class AgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    session_id: str
    last_news: Optional[Dict]
    context: Optional[str]
    current_action: Optional[str]
    extracted_data: Optional[Dict]
    conversation_type: Optional[str]
    user_intent: Optional[str]
    previous_actions: List[str]
    waiting_for: Optional[str]