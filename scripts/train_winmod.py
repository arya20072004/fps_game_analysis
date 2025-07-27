# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/data/VRLMaster_cleaned.csv")

# Preprocess
team_stats = df.groupby("Team")[["ACS", "ADR", "K_D", "Kill_Participation"]].mean()

# Create simulated match data
teams = team_stats.index.tolist()
X = []
y = []

for i in range(len(teams)):
    for j in range(i + 1, len(teams)):
        team_a, team_b = teams[i], teams[j]
        stats_a, stats_b = team_stats.loc[team_a], team_stats.loc[team_b]
        features = list(stats_a.values) + list(stats_b.values)
        X.append(features)

        score_a = stats_a["ACS"] + stats_a["ADR"]
        score_b = stats_b["ACS"] + stats_b["ADR"]
        winner = 1 if score_a > score_b else 0
        y.append(winner)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True),
    "gradient_boost": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train, Evaluate, Save
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

    with open(f"/models/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
