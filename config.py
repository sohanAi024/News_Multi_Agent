import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:root@localhost/news')
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    EMAIL_USER = os.getenv('SMTP_EMAIL')
    EMAIL_PASS = os.getenv('SMTP_PASSWORD')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')