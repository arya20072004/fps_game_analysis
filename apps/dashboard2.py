import streamlit as st
import pandas as pd
import pickle

# Load cleaned data and all models
df = pd.read_csv("/data/VRLMaster_cleaned.csv")

with open("/models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("/models/naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("/models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("/models/gradient_boost_model.pkl", "rb") as f:
    gb_model = pickle.load(f)

# Team Stats for Features
team_stats = df.groupby("Team")[["ACS", "ADR", "K_D", "Kill_Participation"]].mean()
teams = team_stats.index.tolist()

# Model Accuracy Scores 
model_accuracies = {
    "Random Forest": 0.89,
    "Naive Bayes": 0.78,
    "SVM": 0.84,
    "Gradient Boosting": 0.91
}

# Streamlit UI
st.set_page_config(page_title="FPS Game - Winner Predictor", layout="centered")
st.title("🏆 FPS Game - Winner Prediction Dashboard")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Random Forest", "Naive Bayes", "SVM", "Gradient Boosting"]
)

st.sidebar.header("Select Teams")
team_a = st.sidebar.selectbox("Select Team A", teams)
team_b = st.sidebar.selectbox("Select Team B", teams, index=1)

st.subheader(f"Selected Model: {model_choice}")
st.write(f"**Model Accuracy:** {model_accuracies[model_choice]*100:.2f}%")

if team_a == team_b:
    st.warning("Please select two different teams.")
else:
    stats_a = team_stats.loc[team_a]
    stats_b = team_stats.loc[team_b]
    features = list(stats_a.values) + list(stats_b.values)

    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Naive Bayes":
        model = nb_model
    elif model_choice == "SVM":
        model = svm_model
    else:
        model = gb_model

    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]

    winner = team_a if prediction == 1 else team_b
    st.success(f"Predicted Winner: {winner}")

    st.markdown("### 🔍 Confidence Scores")
    st.progress(proba[1] if prediction == 1 else proba[0])
    st.write(f"{team_a}: {proba[1]*100:.2f}%")
    st.write(f"{team_b}: {proba[0]*100:.2f}%")

    st.markdown("---")
    st.markdown("### 📊 Accuracy of All Models")
    for model_name, acc in model_accuracies.items():
        st.write(f"**{model_name}:** {acc*100:.2f}%")
