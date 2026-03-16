# 🎓 Student Exam Performance Indicator

A Machine Learning web application that predicts a student's **Math Score** based on demographic and academic factors. Built with Flask, scikit-learn, and multiple regression models with automated hyperparameter tuning.

---

## 📌 Problem Statement

Understand how a student's performance (math score) is influenced by variables such as:

- Gender
- Race / Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

---

## 🗂️ Project Structure

```
mlproject/
│
├── artifacts/                        # Auto-generated: saved model & preprocessor
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   ├── data/
│   │   └── stud.csv                  # Raw dataset
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   └── model_training.ipynb         # Notebook-level model experiments
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Reads & splits data
│   │   ├── data_transformation.py   # Feature engineering & preprocessing
│   │   └── model_trainer.py         # Model training & evaluation
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py      # Prediction pipeline & CustomData class
│   │
│   ├── exception.py                 # Custom exception handler
│   └── utils.py                     # Utility functions (save/load/evaluate)
│
├── templates/
│   ├── index.html                   # Landing page
│   └── home.html                    # Prediction form & result page
│
├── main.py                          # Flask app entry point
├── setup.py                         # Package setup
└── requirements.txt                 # Python dependencies
```

---

## 🔄 ML Pipeline

```
Raw Data (stud.csv)
      │
      ▼
Data Ingestion          → train.csv, test.csv, data.csv  (artifacts/)
      │
      ▼
Data Transformation     → StandardScaler (numerical) + OneHotEncoder (categorical)
      │                   → preprocessor.pkl  (artifacts/)
      ▼
Model Training          → RandomizedSearchCV on 10 models
      │                   → model.pkl  (artifacts/)
      ▼
Flask Web App           → User input → Preprocessor → Model → Predicted Math Score
```

---

## 🤖 Models Trained

The following regression models are evaluated with **RandomizedSearchCV** hyperparameter tuning:

| Model | Tuned |
|---|---|
| Linear Regression | ✅ |
| Ridge | ✅ |
| Lasso | ✅ |
| K-Nearest Neighbors | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| AdaBoost | ✅ |
| Gradient Boosting | ✅ |
| XGBoost | ✅ |
| CatBoost | ✅ |

The best model (by R² score on the test set) is automatically selected and saved to `artifacts/model.pkl`.

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | male / female |
| `race_ethnicity` | Categorical | group A / B / C / D / E |
| `parental_level_of_education` | Categorical | some high school → master's degree |
| `lunch` | Categorical | standard / free/reduced |
| `test_preparation_course` | Categorical | none / completed |
| `reading_score` | Numerical | Score out of 100 |
| `writing_score` | Numerical | Score out of 100 |
| `math_score` | Numerical | **Target variable** |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlproject.git
cd mlproject
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The `setup.py` makes this project installable as a local package, so `src` imports work correctly.

### 4. Train the model

```bash
python src/components/data_ingestion.py
```

This will:
- Read data from `notebook/data/stud.csv`
- Split into train/test sets and save to `artifacts/`
- Run data transformation and save `preprocessor.pkl`
- Train & tune all models, save the best one as `model.pkl`

### 5. Run the Flask app

```bash
python main.py
```

Visit `http://localhost:5000` in your browser.

---

## 🌐 Web Application

**Landing Page (`/`)** — Introduction to the project with a "Start Prediction" button.

**Prediction Page (`/predict`)** — Fill in the student details form and click **Predict Maths Score** to get the predicted score.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Web Framework:** Flask
- **ML Libraries:** scikit-learn, XGBoost, CatBoost
- **Data Processing:** Pandas, NumPy
- **Visualization (EDA):** Matplotlib, Seaborn
- **Model Persistence:** Pickle

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `notebook/eda.ipynb` | Exploratory Data Analysis — distributions, correlations, score comparisons by gender, ethnicity, parental education, lunch type, and test prep |
| `notebook/model_training.ipynb` | Experimental model training and evaluation at notebook level |

---

## 📁 Key Source Files

| File | Purpose |
|---|---|
| `src/components/data_ingestion.py` | Reads CSV, creates train/test split |
| `src/components/data_transformation.py` | Builds sklearn `Pipeline` + `ColumnTransformer` for preprocessing |
| `src/components/model_trainer.py` | Trains 10 models with `RandomizedSearchCV`, picks the best by R² |
| `src/pipeline/predict_pipeline.py` | Loads saved model & preprocessor for inference; `CustomData` converts form input to DataFrame |
| `src/exception.py` | Custom exception with file name and line number in error messages |
| `src/utils.py` | `save_object`, `load_object`, `evaluate_model` helpers |

---

## ✍️ Author

**Azra**  
Feel free to fork, star ⭐, and raise issues!