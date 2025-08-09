# metro_status_ui.py
import streamlit as st

def display_metro_status_page():
    st.title("🚇 Namma Metro Status")
    st.info("Get real-time operational status and updates for Bangalore Metro lines.")
    
    st.subheader("Line Status (Placeholder)")
    st.write("- **Purple Line:** Running on schedule. (Last updated: 10:30 AM)")
    st.write("- **Green Line:** Minor delays (5 mins) due to technical issue near Jayanagar. (Last updated: 10:35 AM)")
    st.markdown("---")
    st.write("Plan your commute effectively with live Metro updates.")
