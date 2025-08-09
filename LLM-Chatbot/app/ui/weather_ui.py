# weather_ui.py
import streamlit as st

def display_weather_page():
    st.title("🌦️ Bengaluru Weather Updates")
    st.info("Access real-time weather conditions and forecasts for Bangalore.")
    
    st.subheader("Current Weather (Placeholder)")
    st.write("- **Condition:** Partly Cloudy")
    st.write("- **Temperature:** 28°C")
    st.write("- **Humidity:** 65%")
    st.write("- **Wind:** 10 km/h (North-East)")
    st.markdown("---")
    st.write("Forecast for today: Isolated thunderstorms in the afternoon.")
