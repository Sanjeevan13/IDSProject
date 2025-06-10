import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit config
st.set_page_config(page_title="Global Music Trend Predictions", layout="wide")
st.title("🎶 Global Music Trend Predictions using ML")

# Load CSV from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Sanjeevan13/IDSProject/refs/heads/main/Cleaned_Spotify_2024_Global_Streaming_Data.csv"  # Replace with your actual GitHub raw CSV URL
    return pd.read_csv(url)

df = load_data()
st.write("Column names in dataset:", df.columns.tolist())

# EDA Section
st.subheader("🔍 Exploratory Data Analysis")
st.dataframe(df.head())

st.write("Summary Statistics:")
st.dataframe(df.describe())

if "streams" in df.columns:
    top_tracks = df.sort_values("streams", ascending=False).head(10)
    st.bar_chart(top_tracks.set_index("track_name")["streams"])

# ML Section
st.subheader("📊 Machine Learning: Predict Streams")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if "streams" in numeric_cols:
    target = "streams"
    features = [col for col in numeric_cols if col != target]

    if features:
        X = df[features].fillna(0)
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)
        df["Predicted Streams"] = model.predict(X)

        st.write("Actual vs Predicted (sample):")
        st.dataframe(df[[target, "Predicted Streams"]].head())

        fig, ax = plt.subplots()
        sns.scatterplot(x=y, y=df["Predicted Streams"], ax=ax)
        ax.set_xlabel("Actual Streams")
        ax.set_ylabel("Predicted Streams")
        ax.set_title("Actual vs Predicted Streams")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric features found to model.")
else:
    st.warning("⚠️ 'streams' column not found.")

