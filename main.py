import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Global Music Trend Dashboard", layout="wide")

st.title("🎵 Global Music Trend Predictions Dashboard")
st.write("Powered by Spotify Global Streaming Data 2024")

# Load and clean data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Sanjeevan13/IDSProject/refs/heads/main/Cleaned_Spotify_2024_Global_Streaming_Data.csv"  # Replace with your real GitHub raw URL
    df = pd.read_csv(url)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df

df = load_data()
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head())

st.subheader("🧮 Column Names")
st.write(df.columns.tolist())  # Show cleaned column names

# Basic Stats
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Select features and target
features = [
    "monthly_listeners_millions",
    "total_hours_streamed_millions",
    "avg_stream_duration_min",
    "skip_rate_"
]
target = "total_streams_millions"

st.write("Available cleaned columns:", df.columns.tolist())

# Drop rows with missing values in the selected columns
df_model = df[features + [target]].dropna()

# Split data
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.subheader("🧠 Linear Regression Results")
st.write(f"R² Score: `{r2:.4f}`")
st.write(f"RMSE: `{rmse:.4f}`")

# Plot actual vs predicted
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Total Streams (Millions)")
ax.set_ylabel("Predicted Total Streams (Millions)")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# Interactive Prediction
st.subheader("🎯 Predict Total Streams (Millions)")

monthly = st.slider("Monthly Listeners (Millions)", 0.0, 500.0, 100.0)
hours = st.slider("Total Hours Streamed (Millions)", 0.0, 5000.0, 1000.0)
duration = st.slider("Avg Stream Duration (Min)", 1.0, 10.0, 3.0)
skip_rate = st.slider("Skip Rate (%)", 0.0, 100.0, 30.0)

input_data = pd.DataFrame(
    [[monthly, hours, duration, skip_rate]],
    columns=features
)

prediction = model.predict(input_data)[0]
st.success(f"🎧 Predicted Total Streams: `{prediction:.2f} Million`")

st.caption("Data Source: Spotify Global Streaming Data 2024 · Created for WIE2003 Data Science Project")


