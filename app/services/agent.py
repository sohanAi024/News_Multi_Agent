import os
import re
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from app.schemas.chat import AgentState
from app.services.news_tools import NewsTools
from app.services.memory import SessionMemory
from app.services.news_scraper import NewsScraper
from app.services.utils import message_to_dict, dict_to_message
from dotenv import load_dotenv

load_dotenv()

class NewsAgentGraph:
    def __init__(self):
        self.tools = NewsTools()
        self.memory = SessionMemory()
        self.scraper = NewsScraper()
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
        self.llm = ChatMistralAI(api_key=self.mistral_api_key)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the agent conversation graph"""
        g = StateGraph(AgentState)
        
        # Add nodes
        g.add_node("conversation_analysis", self._analyze_conversation)
        g.add_node("search_news", self._search_news)
        g.add_node("summarize_news", self._summarize_news)
        g.add_node("translate", self._translate)
        g.add_node("create_pdf", self._create_pdf)
        g.add_node("send_email", self._send_email)
        g.add_node("follow_up", self._follow_up)
        
        # Set entry point
        g.set_entry_point("conversation_analysis")
        
        # Add conditional edges
        g.add_conditional_edges("conversation_analysis", self._route, {
            "search_news": "search_news",
            "summarize": "summarize_news",
            "translate": "translate",
            "create_pdf": "create_pdf",
            "send_email": "send_email",
            "follow_up": "follow_up"
        })
        
        # Add edges to END
        for node in ["search_news", "summarize_news", "translate", "create_pdf", "send_email", "follow_up"]:
            g.add_edge(node, END)
            
        return g.compile()

    def _analyze_conversation(self, state):
        """Analyze conversation to determine next action"""
        message = state["messages"][-1].content.lower()
        
        if "translate" in message:
            return {**state, "current_action": "translate"}
        elif "summary" in message or "summarize" in message:
            return {**state, "current_action": "summarize"}
        elif "pdf" in message:
            return {**state, "current_action": "create_pdf"}
        elif "email" in message:
            return {**state, "current_action": "send_email"}
        elif "news" in message:
            return {**state, "current_action": "search_news"}
        else:
            return {**state, "current_action": "follow_up"}

    def _route(self, state):
        """Route to appropriate action"""
        return state["current_action"]

    def _search_news(self, state):
        """Search for news"""
        topic = state["messages"][-1].content
        response = self.tools.search_news(topic)
        return {**state, "messages": state["messages"] + [AIMessage(content=response)]}

    def _summarize_news(self, state):
        """Summarize news content"""
        # Find the last news output (look for the news header pattern)
        last_news_msg = None
        for msg in reversed(state["messages"]):
            if (isinstance(msg, AIMessage) and msg.content and 
                not msg.content.startswith(("❗", "❌", "📄", "📧")) and
                ("📰" in msg.content or "🔗" in msg.content)):
                last_news_msg = msg.content
                break
        
        if not last_news_msg:
            return {**state, "messages": state["messages"] + [
                AIMessage(content="❗ No news content found to summarize. Please search for news first.")
            ]}
        
        summary = self.tools.summarize_news(last_news_msg)
        return {**state, "messages": state["messages"] + [AIMessage(content=summary)]}

    def _translate(self, state):
        """Translate news content"""
        user_prompt = state["messages"][-1].content.lower()

        # Find the last news output (look for the news header pattern)
        last_news_msg = None
        for msg in reversed(state["messages"]):
            if (isinstance(msg, AIMessage) and msg.content and 
                not msg.content.startswith(("❗", "❌", "📄", "📧")) and
                ("📰" in msg.content or "🔗" in msg.content)):
                last_news_msg = msg.content
                break

        if not last_news_msg:
            return {**state, "messages": state["messages"] + [
                AIMessage(content="❗ No news content found to translate. Please search for news first.")
            ]}

        # Detect target language
        lang_keywords = {
            "hindi": "Hindi",
            "french": "French", 
            "german": "German",
            "japanese": "Japanese",
            "spanish": "Spanish",
            "chinese": "Chinese",
            "arabic": "Arabic",
            "russian": "Russian"
        }
        
        target_lang = "Hindi"  # default
        for keyword, lang in lang_keywords.items():
            if keyword in user_prompt:
                target_lang = lang
                break

        translated = self.tools.translate_text(last_news_msg, target_lang)
        return {**state, "messages": state["messages"] + [AIMessage(content=translated)]}

    def _create_pdf(self, state):
        """Create PDF from content"""
        # Get the last meaningful AI message content
        content_to_pdf = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("❗") and not msg.content.startswith("📄"):
                content_to_pdf = msg.content
                break
        
        if not content_to_pdf:
            return {**state, "messages": state["messages"] + [AIMessage(content="❗ No content available to create PDF. Please search or get news first.")]}

        # Create PDF with the actual content
        result = self.tools.create_pdf(content_to_pdf, "News Report")
        
        # Store the PDF path in session for potential email sending
        if result.startswith("📄 PDF created successfully:"):
            pdf_filename = result.split(": ")[1]
            session_data = self.memory.get_session(state["session_id"])
            self.memory.update_session(state["session_id"], {"last_pdf_path": pdf_filename})
        
        return {**state, "messages": state["messages"] + [AIMessage(content=result)]}

    def _send_email(self, state):
        """Send email with PDF attachment"""
        user_input = state["messages"][-1].content

        # Extract email address from user input
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
        if not email_match:
            return {**state, "messages": state["messages"] + [AIMessage(content="❗ Please provide a valid email address.")]}

        email_address = email_match.group(0)
        
        # Get the last created PDF path from session
        session_data = self.memory.get_session(state["session_id"])
        pdf_path = session_data.get("last_pdf_path")
        
        if not pdf_path:
            return {**state, "messages": state["messages"] + [AIMessage(content="❗ No PDF found to send. Please create a PDF first.")]}

        result = self.tools.send_email(email_address, pdf_path)
        return {**state, "messages": state["messages"] + [AIMessage(content=result)]}

    def _follow_up(self, state):
        """Provide follow-up options"""
        msg = "What would you like to do next? You can:\n• Search for news\n• Summarize content\n• Translate to another language\n• Create PDF\n• Send email"
        return {**state, "messages": state["messages"] + [AIMessage(content=msg)]}

    def process_message(self, message, session_id):
        """Process incoming message and return response"""
        session_data = self.memory.get_session(session_id)
        history = [dict_to_message(m) for m in session_data.get("messages", [])]
        new_message = HumanMessage(content=message)
        
        state = {
            "messages": history + [new_message],
            "session_id": session_id
        }
        
        result = self.graph.invoke(state)
        response = next((m.content for m in reversed(result["messages"]) if m.type == "ai"), "🤖 I'm here to help!")
        
        # Update session with new messages
        updated_messages = session_data.get("messages", []) + [
            message_to_dict(new_message),
            message_to_dict(AIMessage(content=response))
        ]
        self.memory.update_session(session_id, {"messages": updated_messages})
        
        return response