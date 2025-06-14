import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

st.set_page_config(page_title="Global Music Trend Dashboard", layout="wide")
st.title("🎵 Global Music Trend Predictions Dashboard")
st.write("Powered by Spotify Global Streaming Data 2024")

# Load and clean data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Sanjeevan13/IDSProject/main/Cleaned_Spotify_2024_Global_Streaming_Data.csv"
    df = pd.read_csv(url)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df

df = load_data()

# Show data
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head())

st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Define features and target
features = [
    "monthly_listeners_millions",
    "total_hours_streamed_millions",
    "avg_stream_duration_min",
    "skip_rate_%"
]
target = "total_streams_millions"

# Prepare data
df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor()
}

# Cache trained models
@st.cache_resource
def train_models(X_train, y_train):
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

trained_models = train_models(X_train, y_train)

# Model selection
st.subheader("🧠 Select Model for Prediction")
selected_model_name = st.selectbox("Choose a model", list(trained_models.keys()))
selected_model = trained_models[selected_model_name]

# Make predictions
y_pred = selected_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Selected Model:** `{selected_model_name}`")
st.write(f"R² Score: `{r2:.4f}`")
st.write(f"RMSE: `{rmse:.4f}`")

# Plot
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Total Streams (Millions)")
ax.set_ylabel("Predicted Total Streams (Millions)")
ax.set_title(f"{selected_model_name} - Actual vs Predicted")
st.pyplot(fig)

# User prediction
st.subheader("🎯 Predict Total Streams (Millions)")
monthly = st.slider("Monthly Listeners (Millions)", 0.0, 500.0, 100.0)
hours = st.slider("Total Hours Streamed (Millions)", 0.0, 5000.0, 1000.0)
duration = st.slider("Avg Stream Duration (Min)", 1.0, 10.0, 3.0)
skip_rate = st.slider("Skip Rate (%)", 0.0, 100.0, 30.0)

input_data = pd.DataFrame([[monthly, hours, duration, skip_rate]], columns=features)
predicted_streams = selected_model.predict(input_data)[0]
st.success(f"🎧 Predicted Total Streams ({selected_model_name}): `{predicted_streams:.2f} Million`")

st.caption("Data Source: Spotify Global Streaming Data 2024 · Created for WIE2003 Data Science Project")


