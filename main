import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- Page Config ---
st.set_page_config(page_title="Spotify Trend Predictor", layout="wide")

# --- Title ---
st.title("🎵 Spotify Global Music Trend Predictor")
st.markdown("Predict and visualize global music trends using machine learning.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_data.csv")  # Replace with your cleaned dataset
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter")
region = st.sidebar.selectbox("Select Region", sorted(df['region'].unique()))
date_range = st.sidebar.slider("Select Date Range", 
                                min_value=df['date'].min().date(), 
                                max_value=df['date'].max().date(),
                                value=(df['date'].min().date(), df['date'].max().date()))

# --- Filtered Data ---
filtered_df = df[(df['region'] == region) & 
                 (df['date'].dt.date >= date_range[0]) & 
                 (df['date'].dt.date <= date_range[1])]

# --- Top Songs ---
st.subheader(f"🎧 Top Streamed Songs in {region}")
top_songs = (filtered_df.groupby('track_name')['streams']
                         .sum()
                         .sort_values(ascending=False)
                         .head(10)
                         .reset_index())

chart = alt.Chart(top_songs).mark_bar().encode(
    x=alt.X('streams:Q', title='Total Streams'),
    y=alt.Y('track_name:N', sort='-x', title='Song'),
    color=alt.Color('streams:Q', scale=alt.Scale(scheme='greenblue')),
    tooltip=['track_name', 'streams']
).properties(height=400)

st.altair_chart(chart, use_container_width=True)

# --- Model Section ---
st.subheader("📈 Predict Song Popularity")

model_data = filtered_df[['rank', 'streams']].dropna()
X = model_data[['rank']]
y = model_data['streams']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
st.metric(label="Mean Absolute Error", value=f"{mae:,.0f} streams")

st.write("This basic model predicts a song's stream count based on its rank. For better performance, add more features like artist popularity, genre, and region trends.")

# --- Footer ---
st.markdown("---")
st.markdown("Created for **WIE2003 Introduction to Data Science** | Group Project - 2025")
