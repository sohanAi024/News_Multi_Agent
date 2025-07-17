class SessionMemory:
    def __init__(self): 
        self.sessions = {}
    
    def get_session(self, session_id):
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [], 
                "last_news": None, 
                "context": None,
                "conversation_type": None, 
                "user_intent": None,
                "previous_actions": [], 
                "waiting_for": None,
                "last_pdf_path": None  # Add this to track the last created PDF
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id, updates):
        """Update session with new data"""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
        else:
            self.sessions[session_id] = updates
    
    def clear_session(self, session_id):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        self.sessions.clear()