import streamlit as st
import requests
from datetime import datetime
import re
import os
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change if your API is hosted elsewhere

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().timestamp()}"
if "last_pdf_path" not in st.session_state:
    st.session_state.last_pdf_path = None

# Page setup
st.set_page_config(page_title="Smart News Chat Bot", page_icon="🤖")
st.title("🤖 Smart News Chat Bot")
st.caption("Your AI-powered news assistant with search, summarize, translate, and share capabilities")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"
        st.session_state.last_pdf_path = None
        st.rerun()
    
    if st.button("Refresh News Data"):
        with st.spinner("Updating news database..."):
            response = requests.post(f"{API_BASE_URL}/scrape-news")
            if response.status_code == 200:
                st.success(response.json().get("message", "News updated!"))
            else:
                st.error("Failed to update news")
    
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    - 🔍 Search news (web + database)
    - 📝 Summarize articles
    - 🌍 Translate to multiple languages
    - 📄 Create PDF reports
    - 📧 Email news to anyone
    """)
    
    st.markdown("---")
    st.markdown("### Example Commands")
    st.markdown("""
    - "News about AI developments"
    - "Summarize this"
    - "Translate to French"
    - "Create PDF"
    - "Email to me@example.com"
    """)

    # PDF download section
    if st.session_state.last_pdf_path and os.path.exists(st.session_state.last_pdf_path):
        st.markdown("---")
        st.markdown("### Last Generated PDF")
        with open(st.session_state.last_pdf_path, "rb") as f:
            pdf_data = f.read()
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=os.path.basename(st.session_state.last_pdf_path),
                mime="application/pdf"
            )

# Display chat history
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        content = message["content"]
        
        # Special handling for different message types
        if content.startswith("📄 PDF created successfully:"):
            st.success(content)
            pdf_path = content.split(": ")[1]
            st.session_state.last_pdf_path = pdf_path
            st.rerun()
        elif content.startswith("📧"):
            st.success(content)
        elif content.startswith("❌"):
            st.error(content)
        elif content.startswith("❗"):
            st.warning(content)
        else:
            # Handle news results with better formatting
            if "1. " in content and ("http" in content or "www." in content):
                parts = content.split("\n")
                for part in parts:
                    if part.strip().startswith("http") or part.strip().startswith("www."):
                        st.markdown(f"[🔗 {part}]({part})")
                    else:
                        st.markdown(part)
            else:
                st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me about news or request actions..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat/",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    assistant_response = response_data.get("response", "Sorry, I couldn't process that.")
                    
                    # Handle PDF creation response
                    if assistant_response.startswith("📄 PDF created successfully:"):
                        pdf_path = assistant_response.split(": ")[1]
                        st.session_state.last_pdf_path = pdf_path
                        st.success(assistant_response)
                        st.rerun()
                    elif assistant_response.startswith("📧"):
                        st.success(assistant_response)
                    elif assistant_response.startswith("❌"):
                        st.error(assistant_response)
                    elif assistant_response.startswith("❗"):
                        st.warning(assistant_response)
                    else:
                        # Format news results with better display
                        if "1. " in assistant_response and ("http" in assistant_response or "www." in assistant_response):
                            parts = assistant_response.split("\n")
                            for part in parts:
                                if part.strip().startswith("http") or part.strip().startswith("www."):
                                    st.markdown(f"[🔗 {part}]({part})")
                                else:
                                    st.markdown(part)
                        else:
                            st.markdown(assistant_response)
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    assistant_response = error_msg
                
                # Add to message history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})