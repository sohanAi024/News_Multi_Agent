from sqlalchemy import Column, Integer, Text, DateTime
from datetime import datetime
from pgvector.sqlalchemy import Vector
from .base import Base

class NewsDocument(Base):
    __tablename__ = 'news_documents'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text)
    content = Column(Text)
    category = Column(Text)
    url = Column(Text)
    published_date = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Vector(384))
    
    def __repr__(self):
        return f"<NewsDocument(id={self.id}, title='{self.title[:50]}...')>"