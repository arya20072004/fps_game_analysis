import streamlit as st
import pandas as pd
import pickle

# Load model and team stats
with open("gradient_boost_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("VRLMaster_cleaned.csv")
team_stats = df.groupby("Team")[["ACS", "ADR", "K_D", "Kill_Participation"]].mean()
teams = team_stats.index.tolist()

st.title("🃆 Gradient Boosting - Winner Predictor")

team_a = st.selectbox("Select Team A", teams)
team_b = st.selectbox("Select Team B", teams, index=1)

if team_a == team_b:
    st.warning("Please select two different teams.")
else:
    stats_a = team_stats.loc[team_a]
    stats_b = team_stats.loc[team_b]
    features = list(stats_a.values) + list(stats_b.values)
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]

    st.subheader("Predicted Winner:")
    winner = team_a if prediction == 1 else team_b
    st.success(f"{winner} is predicted to win!")

    st.write("\n**Prediction Confidence:**")
    st.write(f"{team_a}: {proba[1]*100:.2f}%")
    st.write(f"{team_b}: {proba[0]*100:.2f}%")
