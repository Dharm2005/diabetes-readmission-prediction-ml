# 🏥 Diabetes Patient Readmission Prediction - Machine Learning Pipeline

This repository contains the complete Machine Learning workflow used to build the Diabetes Patient Readmission Prediction System.

Unlike the deployment repository, this project focuses on the entire data science pipeline—from understanding the problem to building and evaluating the final machine learning model.

The deployed Streamlit application is available in a separate repository.

---

## 📌 Project Objective

Hospital readmissions are one of the major challenges in healthcare, increasing treatment costs and indicating possible gaps in patient care.

The objective of this project is to predict whether a diabetic patient will be **readmitted within 30 days** after discharge using clinical and hospital encounter data.

Dataset Used:

**Diabetes 130-US Hospitals for Years 1999–2008**

(UCI Machine Learning Repository)

---

# 🔄 Machine Learning Workflow

```
Raw Dataset
      │
      ▼
Problem Definition
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Model Building
      │
      ▼
Model Evaluation
      │
      ▼
Saved Model
```

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

# 📂 Repository Structure

```
Diabetes-Readmission-Prediction-ML/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_problem_definition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_building.ipynb
│
├── models/
│   ├── preprocessing_pipeline.pkl
│   └── final_model.pkl
│
├── requirements.txt
└── README.md
```

---

# 📚 Notebook Overview

| Notebook | Description |
|-----------|-------------|
| 01 | Understanding the business problem, dataset, and project objectives |
| 02 | Data cleaning, handling missing values, removing noisy features, and preparing clean data |
| 03 | Exploratory Data Analysis (EDA) to understand feature distributions and relationships |
| 04 | Feature engineering, preprocessing pipeline creation, and final dataset preparation |
| 05 | Model training, comparison, evaluation, and final model selection |

---

# 🧹 Data Cleaning

The following preprocessing tasks were performed:

- Handling missing values
- Removing noisy and low-impact features
- Cleaning inconsistent categorical values
- Processing diagnosis columns
- Preparing a clean dataset for analysis

---

# 📊 Exploratory Data Analysis

EDA was performed to better understand the dataset before model development.

Major analyses included:

- Target variable distribution
- Numerical feature distributions
- Categorical feature analysis
- Correlation heatmaps
- Clinical insights
- Readmission trend analysis

---

# 🛠 Feature Engineering

Several meaningful features were created to improve model performance.

Examples include:

- `total_diabetes_drugs`
- `oral_med_flag`
- `is_polypharmacy`

Additional preprocessing included:

- Rare category grouping
- One-Hot Encoding
- Numerical feature scaling
- Prevention of data leakage

---

# ⚙️ Preprocessing Pipeline

A reusable preprocessing pipeline was built using **Scikit-learn's ColumnTransformer**.

The pipeline includes:

- Missing value handling
- One-Hot Encoding
- Standard Scaling
- Feature transformation

The fitted preprocessing pipeline was saved for deployment.

---

# 🤖 Model Building

The following machine learning models were trained and compared:

- Logistic Regression ✅ Final Model
- Decision Tree
- Random Forest
- XGBoost

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

The final model was selected based on overall balanced performance and recall for identifying patients at high risk of readmission.

---

# 📁 Generated Artifacts

| Artifact | Description |
|----------|-------------|
| cleaned.csv | Output after data cleaning |
| final_dataset.csv | Dataset after feature engineering |
| preprocessing_pipeline.pkl | Saved preprocessing pipeline |
| final_model.pkl | Trained Logistic Regression model |

---

# 🚀 Reproducing the Project

Clone the repository

```bash
git clone https://github.com/Dharm2005/Diabetes-Readmission-Prediction-ML.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebooks in the following order:

1. Problem Definition
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Building

---

# 📊 Dataset

UCI Machine Learning Repository

**Diabetes 130-US Hospitals Dataset (1999–2008)**

https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

---

# 🔗 Related Repository

This repository focuses on the machine learning workflow.

The deployed Streamlit application is available here:

**Diabetes Patient Readmission Prediction - Streamlit Application**

https://github.com/Dharm2005/Diabetes-Readmission-Predictor

---

# 💡 Future Improvements

- Hyperparameter tuning
- SHAP-based model explainability
- Docker containerization
- CI/CD pipeline
- Automated model retraining
- Model monitoring

---

# 👨‍💻 Author

**Dharm Dobariya**

Machine Learning Project

---

# ⭐ Highlights

✔ Complete Machine Learning Workflow

✔ End-to-End Data Preprocessing

✔ Clinical Feature Engineering

✔ Pipeline-Based Preprocessing

✔ Multiple Model Comparison

✔ Imbalanced Classification Handling

✔ Reproducible Notebook-Based Development

✔ Deployment-Ready Model Generation
