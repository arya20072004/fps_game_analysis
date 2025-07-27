import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load Cleaned Dataset
df = pd.read_csv("VRLMaster_cleaned.csv")

# Step 2: Create Team-level Features
team_stats = df.groupby("Team")[["ACS", "ADR", "K_D", "Kill_Participation"]].mean().reset_index()

# Step 3: Simulate Matchups (Team A vs Team B)
match_data = []
teams = team_stats["Team"].tolist()

for _ in range(500):  # You can adjust the number of matchups
    team_a, team_b = random.sample(teams, 2)
    stats_a = team_stats[team_stats["Team"] == team_a].iloc[0, 1:].values
    stats_b = team_stats[team_stats["Team"] == team_b].iloc[0, 1:].values

    score_a = stats_a[0] + stats_a[1]  # ACS + ADR as a simple skill metric
    score_b = stats_b[0] + stats_b[1]

    winner = 1 if score_a > score_b else 0  # 1 = team_a wins, 0 = team_b wins

    match_data.append(list(stats_a) + list(stats_b) + [winner])

# Step 4: Create DataFrame
columns = ["ACS_A", "ADR_A", "K_D_A", "KP_A", "ACS_B", "ADR_B", "K_D_B", "KP_B", "Winner"]
match_df = pd.DataFrame(match_data, columns=columns)

# Step 5: Train Model
X = match_df.drop("Winner", axis=1)
y = match_df["Winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Save Model
pickle.dump(model, open("winner_model.pkl", "wb"))
print("✅ Model saved as 'winner_model.pkl'")
