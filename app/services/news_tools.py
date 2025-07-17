import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_mistralai.chat_models import ChatMistralAI
from app.models.base import SessionLocal
from app.models.news_document import NewsDocument
from dotenv import load_dotenv
from config import Config
from sqlalchemy import select, func
from langchain.schema import HumanMessage
from langchain_groq.chat_models import ChatGroq
from sqlalchemy import select
from .news_scraper import classify_category_with_mistral

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()

class NewsTools:
    def __init__(self):
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing")

        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        self.session = SessionLocal()

    def normalize_category(self, category: str) -> str:
        if not category:
            return "General"
        category = re.sub(r'^Category:\s*', '', category)
        category = re.sub(r'\(.*?\)', '', category)
        return category.strip()

    def is_ai_related(self, title: str, content: str) -> bool:
        """
        LLM check: Is this news about AI?
        """
        prompt = (
            f"Determine if the following news is specifically related to Artificial Intelligence (AI). "
            f"Return only 'YES' or 'NO'.\n\n"
            f"Title: {title}\n"
            f"Content: {content[:500]}..."
        )
        response = self.llm.invoke(prompt).content.strip().upper()
        return response == "YES"

    def search_news(self, query: str) -> str:
        """
        Advanced AI-news-only search:
        - Detect AI relevance in results.
        - Combine vector similarity + keyword fallback.
        - Summarize top results.
        """
        try:
            # ✅ Step 1: Query embedding
            query_embedding = embedding_model.encode(query).tolist()

            # ✅ Step 2: Search by vector similarity
            stmt = (
                select(
                    NewsDocument,
                    NewsDocument.embedding.cosine_distance(query_embedding).label("distance")
                )
                .order_by("distance")
            )
            rows = self.session.execute(stmt).all()

            if not rows:
                return f"❗ No news found for '{query}'."

            # ✅ Step 3: Filter AI-related articles (strict)
            ai_news = []
            for doc, distance in rows:
                if self.is_ai_related(doc.title, doc.content):
                    ai_news.append((doc, distance))

            if not ai_news:
                return "❗ No AI-related news found for your query."

            # ✅ Step 4: Apply threshold + keyword relevance
            threshold = 0.75
            ranked = []
            keywords = query.lower().split()
            for doc, dist in ai_news:
                score = 1 - dist
                keyword_match = sum(k in (doc.title.lower() + doc.content.lower()) for k in keywords)
                final_score = score + (0.05 * keyword_match)
                ranked.append((doc, final_score))

            # Sort by final score
            ranked.sort(key=lambda x: x[1], reverse=True)
            top_results = ranked[:5]

            # ✅ Step 5: Prepare context for summarization
            news_chunks = []
            for doc, score in top_results:
                normalized_category = self.normalize_category(doc.category)
                news_chunks.append(
                    f"📰 **Title:** {doc.title}\n"
                    f"🔗 **URL:** {doc.url}\n"
                    f"📂 **Category:** {normalized_category}\n"
                    f"📜 **Content:**\n{doc.content}\n"
                    + "-" * 60
                )
            context_text = "\n\n".join(news_chunks)

            # ✅ Step 6: LLM summarization
            prompt = (
                "You are an expert news summarizer. Based on these AI-related articles, answer:\n"
                f"User Query: {query}\n\n"
                "Provide:\n"
                "1. A short, direct answer.\n"
                "2. A concise summary of the main points.\n\n"
                f"Articles:\n{context_text}\n\nAnswer:"
            )
            summary = self.llm.invoke(prompt).content

            return f"🤖 {summary}\n\n📌 Top Matching AI News:\n{context_text}"

        except Exception as e:
            return f"❌ Error during AI news search: {str(e)}"
        finally:
            self.session.close()




    def translate_text(self, text, language="Hindi"):
        try:
            # Enhanced prompt specifically for Hindi translation
            prompt = (
                f"Translate the following English news content to {language} while strictly maintaining:\n"
                "1. All original formatting (emojis, URLs, separators, line breaks)\n"
                "2. Metadata labels (📰, 🔗, 📅) in their original form\n"
                "3. Technical terms and proper nouns (like AI, ChatGPT, Musk) in original English\n"
                "4. Numeric values and dates in original format\n\n"
                "Special Instructions for Hindi:\n"
                "- Use Devanagari script\n"
                "- Keep English technical terms as-is\n"
                "- Maintain news article tone\n\n"
                "Content to translate:\n"
                f"{text}"
            )
            
            # Get translation from Groq with specific parameters for Hindi
            response = self.llm.invoke(
                prompt,
                temperature=0.2,  # Lower temperature for more precise translations
                max_tokens=4000,   # Ensure enough tokens for complete translation
                stop=["###"]       # Stop sequence to prevent runaway generation
            )
            
            # Post-processing to ensure metadata integrity
            translated = response.content
            
            # Preserve all URLs exactly as they were
            url_pattern = r'(https?://[^\s]+)'
            urls = re.findall(url_pattern, text)
            for url in urls:
                translated = translated.replace(url, url)
                
            # Ensure metadata labels remain in original form
            translated = translated.replace("📰", "📰")
            translated = translated.replace("🔗", "🔗")
            translated = translated.replace("📅", "📅")
            
            # Preserve the separator lines
            translated = translated.replace("=" * 60, "=" * 60)
            
            return translated
            
        except Exception as e:
            return f"❌ Translation Error: {str(e)}"

    def summarize_news(self, text):
        try:
            prompt = (
                "Summarize the key points from these news articles. "
                "Keep the original article structure but make each summary concise. "
                "Include the source URLs. Format as:\n\n"
                "📰 [Concise Title]\n"
                "🔗 [URL]\n"
                "📅 [Date]\n"
                "[Bullet point summary]\n\n"
                "Original articles:\n"
                f"{text}"
            )
            return self.llm.invoke(prompt).content
        except Exception as e:
            return f"❌ Error: {e}"

    def create_pdf(self, content: str, title: str = "News Report") -> str:
        try:
            if not content or content.strip() == "":
                return "❌ Error: No content provided for PDF creation."
            
            # Don't clean content - preserve original formatting
            original_content = content
            
            # Check if content contains Devanagari (Hindi) characters
            has_hindi = any(0x0900 <= ord(char) <= 0x097F for char in content)
            
            # Try to use reportlab for better Unicode support
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"news_report_{timestamp}.pdf"
                
                # Create PDF document
                doc = SimpleDocTemplate(filename, pagesize=A4,
                                      rightMargin=72, leftMargin=72,
                                      topMargin=72, bottomMargin=18)
                
                # Register Hindi font if available
                if has_hindi and os.path.exists("NotoSansDevanagari-Regular.ttf"):
                    pdfmetrics.registerFont(TTFont('NotoSansDevanagari', 'NotoSansDevanagari-Regular.ttf'))
                    font_name = 'NotoSansDevanagari'
                else:
                    font_name = 'Helvetica'
                
                # Create styles
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=30,
                    alignment=1,  # Center alignment
                    fontName=font_name
                )
                
                content_style = ParagraphStyle(
                    'CustomContent',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=12,
                    fontName=font_name,
                    leading=14
                )
                
                # Build story
                story = []
                
                # Add title
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 12))
                
                # Add date
                date_str = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                story.append(Paragraph(date_str, content_style))
                story.append(Spacer(1, 20))
                
                # Add content - preserve line breaks and formatting
                lines = original_content.split('\n')
                for line in lines:
                    if line.strip():
                        # Escape HTML entities but preserve the text
                        escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(escaped_line, content_style))
                    else:
                        story.append(Spacer(1, 6))
                
                # Build PDF
                doc.build(story)
                
                # Verify file was created
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    return f"📄 PDF created successfully: {filename}"
                else:
                    return "❌ Error: PDF file was not created properly."
                    
            except ImportError:
                # Fallback to FPDF if reportlab is not available
                return self._create_pdf_fallback(original_content, title)
                
        except Exception as e:
            return f"❌ Error creating PDF: {str(e)}"
    
    def _create_pdf_fallback(self, content: str, title: str) -> str:
        """Fallback PDF creation using FPDF with better Unicode handling"""
        try:
            from fpdf import FPDF
            
            # Create custom FPDF class for better Unicode support
            class UTF8FPDF(FPDF):
                def __init__(self):
                    super().__init__()
                    self.set_auto_page_break(auto=True, margin=15)
                
                def write_utf8(self, h, txt):
                    # Convert text to UTF-8 bytes then to latin-1 for FPDF
                    try:
                        # First try to use the text as-is
                        self.write(h, txt)
                    except:
                        # If that fails, try to encode/decode carefully
                        try:
                            # For mixed content, try to preserve as much as possible
                            encoded = txt.encode('utf-8').decode('utf-8')
                            self.write(h, encoded)
                        except:
                            # Last resort: replace problematic characters
                            safe_txt = txt.encode('ascii', 'ignore').decode('ascii')
                            self.write(h, safe_txt)
            
            pdf = UTF8FPDF()
            pdf.add_page()
            pdf.set_margins(15, 15, 15)
            
            # Add title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, title.encode('latin-1', 'ignore').decode('latin-1'), 0, 1, 'C')
            pdf.ln(10)
            
            # Add date
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.ln(10)
            
            # Add content
            pdf.set_font('Arial', '', 11)
            
            # Split content into lines and process each
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    try:
                        # Try to preserve the original text structure
                        pdf.write_utf8(8, line.strip())
                        pdf.ln(8)
                    except:
                        # If all else fails, write what we can
                        safe_line = ''.join(c for c in line if ord(c) < 256)
                        pdf.write(8, safe_line)
                        pdf.ln(8)
                else:
                    pdf.ln(4)
            
            # Generate filename and save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_report_{timestamp}.pdf"
            pdf.output(filename)
            
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return f"📄 PDF created successfully: {filename}"
            else:
                return "❌ Error: PDF file was not created properly."
                
        except Exception as e:
            return f"❌ Error in fallback PDF creation: {str(e)}"

    def send_email(self, email, pdf_path):
        try:
            if not os.path.exists(pdf_path): 
                return f"❌ PDF file not found: {pdf_path}"
            
            if os.path.getsize(pdf_path) == 0:
                return f"❌ PDF file is empty: {pdf_path}"
            
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_USER
            msg['To'] = email
            msg['Subject'] = "News Report PDF"
            
            # Add email body
            body = "Please find the attached news report PDF."
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(pdf_path)}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(Config.EMAIL_USER, Config.EMAIL_PASS)
            text = msg.as_string()
            server.sendmail(Config.EMAIL_USER, email, text)
            server.quit()
            
            # Clean up - remove PDF after sending
            try:
                os.remove(pdf_path)
            except:
                pass
            
            return f"📧 PDF sent successfully to {email}"
            
        except Exception as e:
            return f"❌ Email Error: {str(e)}"