import streamlit as st
import requests

# === Page Config ===
st.set_page_config(page_title="IEEE Formatter", page_icon="📚")
st.title("📚 AI Chatbot + IEEE Paper Formatter")

# === Flask Server URL (adjust if hosted differently) ===
FLASK_SERVER_URL = "http://localhost:5000/upload"

# === File Upload ===
uploaded_file = st.file_uploader("📄 Upload your paper (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file:
    filetype = uploaded_file.type
    st.success(f"✅ File uploaded: {uploaded_file.name} ({filetype})")

    # === Send to Flask Server ===
    with st.spinner("🔧 Formatting your document in IEEE style..."):
        try:
            files = {"file": uploaded_file}
            response = requests.post(FLASK_SERVER_URL, files=files)

            if response.ok:
                st.success("🎉 Your IEEE formatted paper is ready!")
                st.download_button(
                    "📥 Download Formatted PDF",
                    data=response.content,
                    file_name="ieee_formatted.pdf",
                    mime="application/pdf"
                )
            else:
                st.error(f"❌ Formatter Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Couldn't connect to the formatter service. Is the Flask server running?")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
