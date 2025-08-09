# event_ui.py
import streamlit as st

def display_events_page():
    st.title("📅 Events in Bengaluru")
    st.info("This section will showcase upcoming local festivals, concerts, and cultural events.")
    
    st.subheader("Upcoming Events (Placeholder)")
    st.write("- **Bengaluru Music Fest:** August 15-17, Palace Grounds")
    st.write("- **Karaga Festival:** April (Annual, dates vary), City Center")
    st.write("- **Local Food Market:** Every Saturday, Cubbon Park")
    st.markdown("---")
    st.write("Stay tuned for real-time updates and detailed event listings!")
