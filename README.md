# 🎓 Student Performance — EDA & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualisation-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

An end-to-end data analysis and regression project exploring student assessment scores in mathematics, reading, and writing. It combines exploratory data analysis, baseline regression models, and an interactive Streamlit dashboard.

> **Scope note:** This is an educational portfolio demonstration. Group-level patterns are descriptive, not causal, and the model is not suitable for high-stakes decisions about students.

## ✨ Features

- Data inspection, missing-value checks, duplicate checks, and descriptive summaries
- Score distributions and assessment-score correlation analysis
- Group comparisons by test preparation, lunch type, and parental education
- Three regression baselines: Linear Regression, Decision Tree, and Random Forest
- Held-out evaluation using MAE, RMSE, and R²
- Interactive dashboard filters, plots, sample prediction, and filtered-CSV export
- Preprocessing encapsulated in a scikit-learn pipeline to reduce train/test leakage risk

## 📊 Dashboard preview

![Student Performance dashboard](assets/dashboard-preview.png)

## 🗂️ Repository structure

```text
student-performance-portfolio/
├── app.py
├── data/
│   ├── StudentsPerformance.csv
│   └── README.md
├── notebooks/
│   └── student_performance_modeling.ipynb
├── assets/
│   └── dashboard-preview.png
├── requirements.txt
├── .gitignore
├── LINKEDIN_POST.md
└── README.md
```

## 🚀 Run locally

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. To explore the notebook, run `jupyter notebook` from the project root and open `notebooks/student_performance_modeling.ipynb`.

## 🧪 Modeling approach

1. Validate the expected dataset columns and inspect data quality.
2. Explore distributions, correlations, and selected group summaries.
3. Set **Math Score** as the regression target.
4. Impute and one-hot encode categorical predictors; impute and scale numeric predictors.
5. Use a fixed 80/20 train-test split (`random_state=42`).
6. Compare Linear Regression, Decision Tree, and Random Forest using MAE, RMSE, and R².

The dashboard displays metrics calculated from the included dataset and fixed split; results may vary if the data or modeling choices change.

## ⚠️ Limitations and responsible use

- The dataset has 1,000 records and may not represent other schools, regions, or student populations.
- Reading and writing scores are predictors. They must be available at the intended prediction time; otherwise, the prediction setup would not be valid for that use case.
- A single hold-out split is not evidence of performance on new populations. Cross-validation and external validation would be needed for stronger claims.
- Differences across demographic groups do not establish causation.
- Do not use this demo for admissions, placement, discipline, or decisions affecting educational opportunity.

## 🔭 Possible next steps

- Add cross-validation and hyperparameter tuning
- Review residuals and calibration of prediction errors
- Compare models with and without reading/writing scores
- Add a model card and document dataset provenance
- Deploy the dashboard to Streamlit Community Cloud

## Author

**Nadeem Ahamad**
