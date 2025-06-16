import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

st.set_page_config(page_title="Global Music Trend Dashboard", layout="wide")

st.markdown("""
<div style='display: flex; align-items: center; gap: 10px;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg' width='35'>
    <h1 style='margin: 0;'>Global Music Trend Predictions Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("Dataset: [Spotify Global Streaming Data 2024](https://www.kaggle.com/datasets/atharvasoundankar/spotify-global-streaming-data-2024/data)")

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

# Tabs
overview, model_tab, predict_tab = st.tabs(["📊 Data Overview", "🧠 Model Evaluation", "🎯 Predict Total Streams"])

with overview:
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🧮 Column Names")
    st.write(df.columns.tolist())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# Features and Target
features = [
    "monthly_listeners_millions",
    "total_hours_streamed_millions",
    "avg_stream_duration_min",
    "skip_rate_%"
]
target = "total_streams_millions"

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN Regressor": KNeighborsRegressor()
}

trained_models = {}
model_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    trained_models[name] = model
    model_results.append((name, round(r2, 4), round(rmse, 2)))

results_df = pd.DataFrame(model_results, columns=["Model", "R² Score", "RMSE"])

with model_tab:
    st.subheader("📋 Model Comparison")
    st.dataframe(results_df)

    selected_model_name = st.selectbox("🔧 Select a Model to Visualize", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]
    y_selected_pred = selected_model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2_score(y_test, y_selected_pred):.4f}")
    col2.metric("RMSE", f"{sqrt(mean_squared_error(y_test, y_selected_pred)):.2f}")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_selected_pred, ax=ax)
    ax.set_xlabel("Actual Total Streams (Millions)")
    ax.set_ylabel("Predicted Total Streams (Millions)")
    ax.set_title(f"Actual vs Predicted - {selected_model_name}")
    st.pyplot(fig)

    if hasattr(selected_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': selected_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots()
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_imp)
        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)

with predict_tab:
    st.subheader("🔢 Input Features")
    monthly = st.slider("Monthly Listeners (Millions)", 0.0, 500.0, 100.0)
    hours = st.slider("Total Hours Streamed (Millions)", 0.0, 5000.0, 1000.0)
    duration = st.slider("Avg Stream Duration (Min)", 1.0, 10.0, 3.0)
    skip_rate = st.slider("Skip Rate (%)", 0.0, 100.0, 30.0)

    input_data = pd.DataFrame(
        [[monthly, hours, duration, skip_rate]],
        columns=features
    )

    st.subheader("📊 Predictions from All Models")
    predictions = []
    for name, model in trained_models.items():
        pred = model.predict(input_data)[0]
        predictions.append((name, round(pred, 2)))

    prediction_df = pd.DataFrame(predictions, columns=["Model", "Predicted Streams (Millions)"])
    st.dataframe(prediction_df)

    # Store prediction in session
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Monthly": monthly,
        "Hours": hours,
        "Duration": duration,
        "Skip Rate": skip_rate,
        **{name: pred for name, pred in predictions}
    })

    st.subheader("📚 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.rerun()

    csv = prediction_df.to_csv(index=False)
    st.download_button("📥 Download Prediction", data=csv, file_name="prediction.csv", mime="text/csv")

st.caption("Created for WIE2003 Data Science Project by Group 5")


