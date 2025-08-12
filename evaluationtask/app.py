import streamlit as st
from utils.recommand import get_popular_destinations, filter_hotels
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv('FILE_PATH')

df = load_data()

st.title("🏨 Hotel Search System")

user_input = st.text_input("💬 Enter your preferences (e.g., location, price, amenities):")

# Button
if st.button("🔍 Get Recommendations"):
    if user_input:
        recommendations = filter_hotels(df, user_input)
        if recommendations is not None and not recommendations.empty:
            st.success("✨ Recommended Hotels for You:")
            st.dataframe(recommendations)
        else:
            st.warning("⚠️ No hotels found matching the criteria.")
            st.info("🌍 Here are some popular destinations instead:")
            popular_destinations = get_popular_destinations(df)
            st.dataframe(popular_destinations)
    else:
        st.warning("✏️ Please enter your preferences to get recommendations.")
else:
    st.write("ℹ️ Enter your preferences above and click the button to see hotel recommendations.")
