import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚀 Aerospace AI Journey — Flight Delay Explorer")
st.write("Week 1 prototype — predicting delays with your first ML model")

df = pd.read_csv('data/flight_delay_data.csv')

st.subheader("Raw Data Preview")
st.dataframe(df.head())

st.subheader("Delay Distribution")
fig, ax = plt.subplots()
df['Departure_Delay_min'].hist(bins=50, ax=ax)
st.pyplot(fig)

# Add a simple predictor later — we'll expand this next week
st.success("✅ Deployed! Your first AI product is live.")