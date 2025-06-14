import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

st.set_page_config(page_title="Global Music Trend Dashboard", layout="wide")

st.title("🎵 Global Music Trend Predictions Dashboard")
st.write("Powered by Spotify Global Streaming Data 2024")

# Load and clean data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Sanjeevan13/IDSProject/refs/heads/main/Cleaned_Spotify_2024_Global_Streaming_Data.csv"
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
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head())

# Columns
st.subheader("🧮 Column Names")
st.write(df.columns.tolist())

# Basic stats
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Features & Target
features = [
    "monthly_listeners_millions",
    "total_hours_streamed_millions",
    "avg_stream_duration_min",
    "skip_rate_%"
]
target = "total_streams_millions"

df_model = df[features + [target]].dropna()

# Train/Test Split
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
}

# Try to add XGBoost (if installed)
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(random_state=42)
except ImportError:
    st.warning("⚠️ XGBoost not installed. Skipping it.")

model_metrics = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    model_metrics[name] = {"r2": r2, "rmse": rmse}
    predictions[name] = y_pred

# Show model results
st.subheader("🧠 Model Performance Comparison")
st.write("Comparison of R² and RMSE for each model:")

metric_df = pd.DataFrame(model_metrics).T
st.dataframe(metric_df.style.format({"r2": "{:.4f}", "rmse": "{:.4f}"}))

# Plot actual vs predicted for all models
st.subheader("📉 Actual vs Predicted (All Models)")
fig, ax = plt.subplots(figsize=(10, 6))
for name, y_pred in predictions.items():
    sns.scatterplot(x=y_test, y=y_pred, label=name, alpha=0.6)
ax.set_xlabel("Actual Total Streams (Millions)")
ax.set_ylabel("Predicted Total Streams (Millions)")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# Interactive prediction
st.subheader("🎯 Predict Total Streams")

monthly = st.slider("Monthly Listeners (Millions)", 0.0, 500.0, 100.0)
hours = st.slider("Total Hours Streamed (Millions)", 0.0, 5000.0, 1000.0)
duration = st.slider("Avg Stream Duration (Min)", 1.0, 10.0, 3.0)
skip_rate = st.slider("Skip Rate (%)", 0.0, 100.0, 30.0)

input_df = pd.DataFrame([[monthly, hours, duration, skip_rate]], columns=features)

st.write("Select a model for prediction:")
selected_model_name = st.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

prediction = selected_model.predict(input_df)[0]
st.success(f"🎧 Predicted Total Streams with {selected_model_name}: `{prediction:.2f} Million`")

st.caption("Data Source: Spotify Global Streaming Data 2024 · Created for WIE2003 Data Science Project")


st.caption("Data Source: Spotify Global Streaming Data 2024 · Created for WIE2003 Data Science Project")


