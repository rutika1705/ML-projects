# 🎓 Student Math Score Predictor — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-green?style=flat-square&logo=flask)
![CatBoost](https://img.shields.io/badge/CatBoost-Best%20Model-yellow?style=flat-square)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.88-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A production-ready ML pipeline that predicts student math scores based on demographic and academic factors — from raw data ingestion to a deployed Flask web application.

---

## 📌 Problem Statement

Student performance is influenced by gender, ethnicity, parental education, lunch type, and test preparation. This project builds an end-to-end ML system to **predict a student's math score**, enabling early identification of students who may need academic support.

---

## 🏗️ Project Architecture
```
Data Ingestion → Data Transformation → Model Training → Evaluation → Flask Web App
     ↓                  ↓                    ↓               ↓             ↓
  Raw CSV        Feature Engineering     7 ML Models    Best Model    Prediction UI
                 + Preprocessing        + Hyperparam    (CatBoost)
                                           Tuning
```

---

## 📊 Model Comparison

| Model | R² Score |
|-------|----------|
| **CatBoost Regressor** ✅ | **0.88** |
| XGBoost Regressor | ~0.85 |
| Gradient Boosting | ~0.84 |
| Random Forest | ~0.83 |
| Linear Regression | ~0.78 |
| AdaBoost Regressor | ~0.75 |
| Decision Tree | ~0.70 |

> ✅ **CatBoost** was auto-selected as the best model — explains **88% of variance** in student math scores.

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Models | CatBoost, XGBoost, Scikit-learn |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS (Jinja2) |

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

---

## 📈 Input Features

| Feature | Description |
|---------|-------------|
| `gender` | Student's gender |
| `race/ethnicity` | Ethnic group |
| `parental_level_of_education` | Parent's highest education |
| `lunch` | Standard or free/reduced |
| `test_preparation_course` | Completed or none |
| `reading_score` | Reading score |
| `writing_score` | Writing score |

**Target:** `math_score`

---

## 👤 Author

**Rutika**
- 🔗 [GitHub](https://github.com/rutika1705)
- 💼 [LinkedIn](https://www.linkedin.com/in/rutika-tharali/) ← 

---

## 📄 License

MIT License
