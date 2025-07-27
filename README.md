# 🎮 FPS Game Analytics & Winner Prediction

This project offers a complete toolkit for analyzing competitive FPS game data. It features an exploratory data analysis (EDA) dashboard to visualize player and team statistics, alongside a machine learning-powered application to predict match outcomes.

---

## 🚀 Key Features

- **EDA Dashboard**: An interactive interface to explore various game statistics.
  - Displays the top 10 teams by average ACS (Average Combat Score).
  - Visualizes the ACS distribution for all players.
  - Includes a correlation heatmap for all numerical stats.
  - Shows a leaderboard of top players by K/D ratio.
- **Winner Prediction Dashboard**: A tool to predict the winner between two teams using a choice of four different machine learning models.
  - Allows selection from Random Forest, Naive Bayes, SVM, and Gradient Boosting models.
  - Displays the selected model's accuracy and confidence scores for its prediction.
- **Data Processing & Training**: Includes scripts to process raw data and train the machine learning models from scratch.

---

## 📂 Project Structure

```

├── .gitignore
├── README.md
├── requirements.txt
├── data
│ ├── FPS_Game_Cleaned.csv
│ └── VRLMaster_cleaned.csv
├── models
│ ├── gradient_boost_model.pkl
│ ├── naive_bayes_model.pkl
│ ├── random_forest_model.pkl
│ └── svm_model.pkl
├── scripts
│ ├── FPSgame.py
│ └── train_winmod.py
└── apps
├── dashboard.py
└── dashboard2.py

```

---

## 🤖 Machine Learning Models

The prediction dashboard uses four machine learning models. Their accuracy, as evaluated on the test set, is listed below:

| Model              | Accuracy |
|-------------------|----------|
| Gradient Boosting | 91.00%   |
| Random Forest     | 89.00%   |
| SVM               | 84.00%   |
| Naive Bayes       | 78.00%   |

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder-name>
   ```

2. **Install dependencies:**

   Ensure you have Python 3.8+ installed. Then, install the required packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♀️ Usage

Follow the steps below to run the data pipeline and launch the applications.

### 1. Data Processing and Model Training (Optional)

If you wish to re-process the data or re-train the models, run the scripts from the `scripts/` directory:

- **To process the raw data:**
  ```bash
  python scripts/FPSgame.py
  ```

- **To train the ML models:**
  This will generate the `.pkl` files and save them in the `models/` directory.

  ```bash
  python scripts/train_winmod.py
  ```

### 2. Running the Dashboards

The main applications are located in the `apps/` directory.

- **To launch the EDA Dashboard:**
  ```bash
  streamlit run apps/dashboard.py
  ```

- **To launch the Winner Prediction Dashboard:**
  ```bash
  streamlit run apps/dashboard2.py
  ```

Once an application is running, open your web browser and navigate to the local URL provided by Streamlit (e.g., `http://localhost:8501`).

---

## 📌 Notes

- Make sure all datasets (`.csv`) and model files (`.pkl`) are present in their respective directories.
- Modify the scripts as needed if you're using a different dataset schema or adding features.
- This project is designed for demonstration and learning purposes but can be extended further with real-time match data, player tracking, or deeper analytics.

---

## 📧 Contact

For suggestions or queries, feel free to reach out via [GitHub Issues](https://github.com/your-username/your-repo-name/issues).

---
