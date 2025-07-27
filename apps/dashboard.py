import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("/data/VRLMaster_cleaned.csv")

st.title("🎮 FPS Game Data Dashboard")

# Team-wise Average ACS
st.subheader("Top 10 Teams by Average ACS")
top_teams = df.groupby("Team")["ACS"].mean().sort_values(ascending=False).head(10)
st.bar_chart(top_teams)

# ACS Distribution
st.subheader("ACS Distribution")
fig, ax = plt.subplots()
sns.histplot(df["ACS"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure for better visibility
sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={"label": "Correlation Coefficient"},
    ax=ax
)
st.pyplot(fig)

# Kill Participation vs ACS
st.subheader("Kill Participation vs ACS")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Kill_Participation", y="ACS", hue="Team", ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Team")  # Move legend outside plot
st.pyplot(fig)

# Player Leaderboard
st.subheader("Top Players by K/D Ratio")
top_players = df.nlargest(10, "K_D")[["Ign", "K_D"]]
st.dataframe(top_players.set_index("Ign"))
