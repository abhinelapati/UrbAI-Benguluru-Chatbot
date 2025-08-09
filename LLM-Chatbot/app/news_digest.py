# news_digest.py
import streamlit as st

def display_news_page():
    st.title("📰 Bengaluru News Digest")
    st.info("Get the latest curated news and headlines relevant to Bangalore city here.")
    
    st.subheader("Latest Headlines (Placeholder)")
    st.write("- **New Metro Line Extension Approved:** Construction to begin next quarter.")
    st.write("- **BBMP Launches Cleanliness Drive:** Focus on waste management in central areas.")
    st.write("- **Startup Ecosystem Flourishes:** Bengaluru attracts record investment in Q2.")
    st.markdown("---")
    st.write("Powered by trusted local news sources.")
