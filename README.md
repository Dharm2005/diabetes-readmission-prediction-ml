# 🏥 Diabetes Patient Readmission Prediction - Machine Learning Pipeline

An end-to-end Machine Learning project focused on predicting **30-day hospital readmission among diabetic patients** using clinical and hospital encounter data.

This repository contains the complete data science workflow, from problem definition and data cleaning to exploratory data analysis, feature engineering, model development, and evaluation.

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

The Streamlit prediction application is maintained in a separate repository.

---

## 📌 Project Objective

Hospital readmissions are an important healthcare outcome associated with patient management, treatment effectiveness, and healthcare resource utilization.

The objective of this project is to develop a supervised machine learning model that predicts whether a diabetic patient will be readmitted to a hospital within 30 days after discharge.

The original `readmitted` target variable contains three categories:

| Original Value | Meaning |
|---|---|
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days |
| `NO` | No recorded readmission |

For this project, the problem is formulated as a **binary classification task**:

- `1` – Readmitted within 30 days (`<30`)
- `0` – Not readmitted within 30 days (`>30` or `NO`)

The dataset is imbalanced, with approximately 11.16% of encounters belonging to the positive class.

---

## 📊 Dataset Information

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

**Source:** UCI Machine Learning Repository

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

| Property | Description |
|---|---|
| Total encounters | 101,766 |
| Original columns | 50 |
| Hospitals | 130 US hospitals |
| Time period | 1999–2008 |
| Target variable | `readmitted` |
| Machine learning task | Binary Classification |
| Positive-class prevalence | Approximately 11.16% |

The dataset contains demographic, clinical, laboratory, medication, diagnosis, and hospital encounter information.

It also contains missing values, repeated encounters for some patients, and categorical variables with high cardinality.

---

## 🔄 Project Workflow

The project follows a structured machine learning workflow:

```text
Raw Dataset
     |
     ▼
01. Problem Definition
     |
     ▼
02. Data Cleaning
     |
     ▼
03. Exploratory Data Analysis
     |
     ▼
04. Feature Engineering
     |
     ▼
05. Preprocessing Pipeline
     |
     ▼
06. Model Training & Comparison
     |
     ▼
07. Validation-Based Model Selection
     |
     ▼
08. Test Evaluation
     |
     ▼
Saved Research Models & Thresholds
```

---

## 📚 Notebook Overview

The project is organized into five Jupyter notebooks.

| Notebook | Description |
|---|---|
| `01_problem_definition.ipynb` | Defines the prediction problem, dataset, target variable, and machine learning objective. |
| `02_cleaning.ipynb` | Handles missing values, categorical inconsistencies, diagnosis codes, and dataset cleaning. |
| `03_eda.ipynb` | Performs exploratory data analysis, visualizations, feature distributions, correlation analysis, and readmission analysis. |
| `04_feature_engineering.ipynb` | Creates engineered features, prepares the binary target, and generates the full and eligible-discharge datasets. |
| `05_model_building.ipynb` | Performs patient-level data splitting, preprocessing, model comparison, threshold selection, and final evaluation. |

---

## 🧹 1. Data Cleaning

The raw dataset required extensive cleaning before exploratory analysis and model development.

Major cleaning operations included:

- Identified missing values across all dataset columns.
- Dropped `weight` because approximately 96.86% of its values were missing.
- Dropped `max_glu_serum` because approximately 94.75% of its values were missing.
- Retained `payer_code` and represented missing values as `Unknown`.
- Handled missing values in `race`, `medical_specialty`, and diagnosis columns.
- Reduced categorical cardinality where appropriate.
- Used the original `IDS_mapping.csv` file to convert admission, discharge, and admission-source identifiers into descriptive categories.
- Grouped diagnosis codes into broader ICD-9-based medical categories.
- Standardized unknown and unavailable categorical values.
- Preserved `patient_nbr` for patient-level data splitting.

The cleaned dataset contains:

**101,766 rows × 47 columns**

The patient identifier is retained for splitting but is never used as a model predictor.

---

## 📊 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of the data and investigate relationships between patient characteristics, hospital encounters, and readmission outcomes.

### Numerical Analysis

The analysis included:

- Histograms and KDE plots.
- Boxplots and IQR-based outlier detection.
- Mean and median comparisons.
- Correlation analysis using heatmaps.
- Analysis of hospital stays, procedures, medications, and previous visits.

### Categorical Analysis

The analysis included:

- Category frequency distributions.
- Cross-tabulations against the target variable.
- Count plots comparing categorical variables with readmission outcomes.
- Analysis of age groups, gender, race, admission types, discharge dispositions, medication changes, and insulin status.

### Readmission Analysis

Additional analysis examined:

- Age versus readmission.
- Insulin status versus readmission.
- Hospital stay duration versus readmission.
- Previous inpatient visits versus readmission.

EDA observations were used to understand the dataset and guide subsequent feature engineering.

---

## 🛠 3. Feature Engineering

Feature engineering was performed to simplify medication information and create additional variables representing treatment patterns and encounter characteristics.

### Target Transformation

The original three-class target was converted into a binary target:

```python
df['readmitted_binary'] = df['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0
)
```

### Engineered Features

| Feature | Description |
|---|---|
| `total_diabetes_drugs` | Counts diabetes medication indicators recorded as anything other than `No`. |
| `insulin_flag` | Binary indicator of whether insulin was recorded. |
| `oral_med_flag` | Indicates whether any non-insulin diabetes medication was recorded. |
| `high_medication_count_10plus` | Indicates whether the hospital encounter involved 10 or more medications. |

### Medication Transformation

The original medication columns contained the following values:

- `No`
- `Steady`
- `Up`
- `Down`

They were converted into binary indicators:

| Original Value | Encoded Value |
|---|---:|
| `No` | 0 |
| `Steady` | 1 |
| `Up` | 1 |
| `Down` | 1 |

This transformation focuses on whether a medication was recorded rather than its dosage-change status.

The original individual medication columns were removed after the engineered features were created.

### Additional Transformations

- Medication changes: `No = 0`, `Ch = 1`.
- Diabetes medication indicator: `No = 0`, `Yes = 1`.
- Age intervals were converted into approximate midpoint values.
- `A1Cresult` was retained as a categorical variable.
- `payer_code` was retained as a categorical predictor.

The final engineered dataset contains:

**101,766 rows × 28 columns**

This includes 26 model predictors, `patient_nbr` for grouping, and `readmitted_binary` as the target.

---

## 🧪 4. Cohort Definition and Sensitivity Analysis

Two analysis cohorts were prepared.

### Full Cohort

The full cohort contains all 101,766 hospital encounters.

It includes encounters involving death-related and hospice discharges.

### Eligible-Discharge Cohort

A separate eligible-discharge cohort was created by excluding encounters with death-related and hospice discharge dispositions.

| Property | Value |
|---|---:|
| Full cohort encounters | 101,766 |
| Excluded encounters | 2,423 |
| Eligible-discharge encounters | 99,343 |
| Unique patients in eligible cohort | 69,990 |
| Eligible positive-class prevalence | Approximately 11.39% |

The original full dataset remains unchanged.

The eligible-discharge cohort is used as the primary analysis population, while the full cohort is retained as a secondary analysis for comparison.

Both cohorts use the same patient-level train, validation, and test assignments.

This provides a more controlled comparison of model performance across the two cohort definitions.

---

## ⚙️ 5. Preprocessing Pipeline

A reusable preprocessing workflow was implemented using Scikit-learn's `ColumnTransformer` and `Pipeline`.

The model uses **26 predictors**:

- 15 numerical features.
- 11 categorical features.

### Numerical Preprocessing

A numerical pipeline was constructed using:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### Categorical Preprocessing

A categorical pipeline was constructed using:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)`

Infrequent categories are grouped based on training data.

This ensures that category-frequency handling and preprocessing are learned from the training partition rather than the complete dataset.

### Patient-Level Data Splitting

The dataset contains multiple hospital encounters for some patients.

To avoid placing encounters belonging to the same patient in different partitions, `patient_nbr` is used as a grouping variable.

Three partitions are created:

1. Training set – model fitting.
2. Validation set – model comparison and threshold selection.
3. Test set – final independent evaluation.

Patient identifiers are excluded from all model predictors.

The full and eligible-discharge cohorts reuse the same patient-level assignments.

---

## 🤖 6. Model Building and Evaluation

The following models were evaluated:

- Logistic Regression
- Random Forest
- Decision Tree
- Dummy Classifier (prior baseline)

Class imbalance was addressed using class weighting for the candidate tree-based models and Logistic Regression.

### Model Selection Strategy

Model selection was based on **Validation Average Precision**, which is useful for evaluating positive-class performance in imbalanced classification problems.

The classification threshold was then selected using the validation F1-score.

The test set was kept separate from model and threshold selection.

### Selected Model

**Logistic Regression** was selected for both the full and eligible-discharge cohorts.

It achieved the highest validation Average Precision among the evaluated candidate models in both experiments.

The eligible-discharge cohort is the primary research model, while the full-cohort model is retained as a secondary analysis.

---

## 📈 7. Final Model Results

The following results are from the final held-out test evaluations.

### Primary Model — Eligible-Discharge Cohort

**Selected Model:** Logistic Regression  
**Decision Threshold:** 0.57

| Metric | Test Result |
|---|---:|
| Accuracy | 77.91% |
| ROC-AUC | 0.6585 |
| Average Precision | 0.2032 |
| Precision | 20.51% |
| Recall | 35.95% |
| F1-score | 26.12% |
| Specificity | 83.02% |

### Secondary Model — Full Cohort

**Selected Model:** Logistic Regression  
**Decision Threshold:** 0.56

| Metric | Test Result |
|---|---:|
| Accuracy | 76.46% |
| ROC-AUC | 0.6641 |
| Average Precision | 0.2025 |
| Precision | 19.70% |
| Recall | 39.19% |
| F1-score | 26.22% |
| Specificity | 80.91% |

### Cohort Comparison

| Metric | Full Cohort | Eligible-Discharge Cohort |
|---|---:|---:|
| Encounters | 101,766 | 99,343 |
| Positive-class rate | 11.16% | 11.39% |
| Selected Model | Logistic Regression | Logistic Regression |
| Validation Average Precision | 0.2253 | 0.2261 |
| Threshold | 0.56 | 0.57 |
| Test ROC-AUC | 0.6641 | 0.6585 |
| Test Average Precision | 0.2025 | 0.2032 |
| Test Precision | 0.1970 | 0.2051 |
| Test Recall | 0.3919 | 0.3595 |
| Test F1-score | 0.2622 | 0.2612 |
| Test Specificity | 0.8091 | 0.8302 |
| Test Accuracy | 0.7646 | 0.7791 |

The results show that excluding death-related and hospice discharge encounters produces small changes in predictive performance and the precision-recall balance, rather than a substantial overall improvement.

These findings are specific to the dataset, cohort definitions, and evaluation procedure used in this project.

---

## 💾 8. Saved Model Artifacts

The trained research pipelines and their corresponding decision thresholds are saved separately.

| File | Description |
|---|---|
| `research_final_model.pkl` | Fitted Logistic Regression pipeline for the full cohort. |
| `research_threshold.pkl` | Validation-selected classification threshold for the full cohort. |
| `research_final_model_eligible_discharge.pkl` | Fitted Logistic Regression pipeline for the eligible-discharge cohort. |
| `research_threshold_eligible_discharge.pkl` | Validation-selected classification threshold for the eligible-discharge cohort. |

Each research model is a complete Scikit-learn pipeline containing its fitted preprocessing and classification steps.

The older `final_model.pkl` and `preprocessing_pipeline.pkl` files are retained as legacy project artifacts and are not used by the updated research application.

---

## 📂 Repository Structure

```text
diabetes-readmission-prediction-ml/
│
├── data/
│   ├── 1. raw/
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   │
│   ├── 2. interim/
│   │   ├── clean.csv
│   │   └── cleaned.csv
│   │
│   └── 3. processed/
│       ├── final_dataset.csv
│       └── eligible_discharge_dataset.csv
│
├── models/
│   ├── final_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── research_final_model.pkl
│   ├── research_threshold.pkl
│   ├── research_final_model_eligible_discharge.pkl
│   └── research_threshold_eligible_discharge.pkl
│
├── Notebooks/
│   ├── 01_problem_definition.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_building.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 Reproducing the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Dharm2005/diabetes-readmission-prediction-ml.git

cd diabetes-readmission-prediction-ml

git switch research-ready-pipeline
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks in Order

Open the `Notebooks` directory and execute:

1. `01_problem_definition.ipynb`
2. `02_cleaning.ipynb`
3. `03_eda.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_building.ipynb`

The notebooks use relative paths to read the datasets and save the generated datasets and model artifacts.

---

## ⚠️ Limitations

- The dataset represents hospital encounters from 130 US hospitals during 1999–2008 and may not reflect current healthcare practices.
- Readmission outcomes are limited to records available within the dataset; no recorded readmission does not necessarily mean a patient was never readmitted elsewhere.
- The positive class is relatively small, making prediction challenging.
- The model's precision and recall indicate that it produces both false-positive and false-negative predictions.
- The analysis demonstrates predictive associations and does not establish causal relationships.
- The model has not been clinically validated and should not be used for real-world medical decisions.

---

## 🔗 Related Repository

The interactive Streamlit application using the eligible-discharge research model is maintained separately:

**Diabetes Patient Readmission Prediction – Streamlit Application**

https://github.com/Dharm2005/Diabetes-Readmission-Predictor

---

## 💡 Future Improvements

- Further model optimization and hyperparameter tuning.
- Model explainability using SHAP.
- Additional feature selection and ablation experiments.
- Robustness analysis across patient groups and cohort definitions.
- External validation using independent healthcare datasets.
- Improved model calibration and threshold analysis.
- Reproducible experiment tracking.

---

## 👨‍💻 Author

**Dharm Dobariya**

Machine Learning Project

---

## ⭐ Project Highlights

- Complete end-to-end machine learning workflow.
- Extensive data cleaning and exploratory analysis.
- Domain-informed feature engineering.
- Patient-level grouped evaluation.
- Separate full and eligible-discharge cohort analysis.
- Reusable Scikit-learn preprocessing pipelines.
- Model comparison using validation Average Precision.
- Validation-based classification threshold selection.
- Independent test-set evaluation.
- Saved research models for application integration.
