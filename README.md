# 🎓 Student Math Score Predictor — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-green?style=flat-square&logo=flask)
![LinearRegression](https://img.shields.io/badge/Linear%20Regression-Best%20Model-orange?style=flat-square)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.88-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A production-ready ML pipeline that predicts student math scores based on demographic and academic factors — from raw data ingestion to a deployed Flask web application.

---

## 🌐 Live Demo

👉 [Try the App Here](https://ml-projects-o06y.onrender.com)

---

## 📌 Problem Statement

Student performance is influenced by gender, ethnicity, parental education, lunch type, and test preparation. This project builds an end-to-end ML system to **predict a student's math score**, enabling early identification of students who may need academic support.

---

## 🏗️ Project Architecture
```
Data Ingestion → Data Transformation → Model Training → Evaluation → Flask Web App
     ↓                  ↓                    ↓               ↓             ↓
  Raw CSV        Feature Engineering     7 ML Models    Best Model    Prediction UI
                 + Preprocessing        + Hyperparam  (Linear Reg.)
                                           Tuning
```

---

## 📊 Model Comparison

| Model | R² Score |
|-------|----------|
| **Linear Regression** ✅ | **0.88** |
| CatBoost Regressor | ~0.85 |
| Gradient Boosting | ~0.84 |
| Random Forest | ~0.83 |
| XGBoost Regressor | ~0.82 |
| AdaBoost Regressor | ~0.75 |
| Decision Tree | ~0.70 |

> ✅ **Linear Regression** was auto-selected as the best model with R² = **0.88**, demonstrating that student math scores have strong linear relationships with the input features.

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Models | Scikit-learn, CatBoost, XGBoost |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS (Jinja2) |
| Deployment | Render |

---

## 📁 Project Structure
```
ML-projects/
│
├── artifacts/               # Saved model & preprocessor (.pkl)
├── notebook/                # EDA & training notebooks
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utilis.py
├── templates/               # Flask HTML templates
├── app.py
├── requirements.txt
└── setup.py
```

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/rutika1705/ML-projects.git
cd ML-projects
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open → `http://localhost:5000`

---

## 🔍 Key Features

- ✅ Modular pipeline — each ML step is a separate reusable component
- ✅ Automated model selection — trains 7 models, picks best automatically
- ✅ Hyperparameter tuning via GridSearchCV
- ✅ Custom exception handling & logging
- ✅ Flask web app for live predictions
- ✅ Deployed on Render — accessible anywhere

---

## 📈 Input Features

| Feature | Description |
|---------|-------------|
| `gender` | Student's gender |
| `race/ethnicity` | Ethnic group |
| `parental_level_of_education` | Parent's highest education |
| `lunch` | Standard or free/reduced |
| `test_preparation_course` | Completed or none |
| `reading_score` | Reading score (0–100) |
| `writing_score` | Writing score (0–100) |

**Target:** `math_score`

---

## 👤 Author

**Rutika**
- 🔗 [GitHub](https://github.com/rutika1705)
- 💼 [LinkedIn](https://www.linkedin.com/in/rutika-tharali) ← paste your LinkedIn URL here

---

## 📄 License

MIT License
