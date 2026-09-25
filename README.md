# Student Performance — Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An end-to-end student assessment analysis project exploring math, reading, and writing scores, comparing score patterns across selected student attributes, and evaluating baseline machine-learning models for estimating math scores.

The project includes a cleaned Jupyter Notebook, an interactive Streamlit dashboard, model comparison using a held-out test set, and an exploratory prediction form.

## Project Highlights

- Exploratory analysis of student assessment scores
- Data-quality checks and descriptive statistics
- Interactive score distributions and correlation heatmap
- Group comparisons by test preparation, lunch type, and parental education
- Linear Regression, Decision Tree, and Random Forest regression baselines
- Model evaluation using MAE, RMSE, and R²
- Interactive Streamlit dashboard with filters and downloadable filtered data
- Demonstration form for estimating a math score
- Reproducible preprocessing and modeling pipelines

## Dataset

The project uses `data/StudentsPerformance.csv`, containing **1,000 records and 8 columns**:

- Categorical attributes: Gender, Race/Ethnicity, Parental Level Of Education, Lunch, and Test Preparation Course
- Assessment scores: Math Score, Reading Score, and Writing Score

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Student-Performance-Predictive-Modeling/
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
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows (Git Bash):**
```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard includes interactive filters, score distributions, score correlations, group comparisons, model evaluation metrics, a sample prediction form, and CSV export for the filtered records.

## Run the Notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open `notebooks/student_performance_modeling.ipynb` and run the cells in order.

## Model Workflow

1. Load and validate the dataset
2. Review missing values, duplicates, and descriptive statistics
3. Explore score distributions and relationships
4. Encode categorical fields using a preprocessing pipeline
5. Split the data into training and test sets
6. Train Linear Regression, Decision Tree, and Random Forest regressors
7. Compare models using MAE, RMSE, and R²
8. Explore predictions in the dashboard

## Dashboard Preview

### Dashboard — Overview
![Student Performance Dashboard Preview 2](assets/dashboard-preview%20%282%29.png)

### Dashboard — View 3
![Student Performance Dashboard Preview 3](assets/dashboard-preview%20%283%29.png)

### Dashboard — View 4
![Student Performance Dashboard Preview 4](assets/dashboard-preview%20%284%29.png)

### Dashboard — View 5
![Student Performance Dashboard Preview 5](assets/dashboard-preview%20%285%29.png)

### Dashboard — View 6
![Student Performance Dashboard Preview 6](assets/dashboard-preview%20%286%29.png)

### Dashboard — View 7
![Student Performance Dashboard Preview 7](assets/dashboard-preview%20%287%29.png)

### Dashboard — View 8
![Student Performance Dashboard Preview 8](assets/dashboard-preview%20%288%29.png)

### Dashboard — View 9
![Student Performance Dashboard Preview 9](assets/dashboard-preview%20%289%29.png)

### Dashboard — View 10
![Student Performance Dashboard Preview 10](assets/dashboard-preview%20%2810%29.png)

### Dashboard — View 11
![Student Performance Dashboard Preview 11](assets/dashboard-preview%20%2811%29.png)

## Important Limitations

- This is an educational portfolio project, not a validated student assessment system.
- Reading and writing scores are model inputs. They should only be used when available at the time the math score is being estimated.
- Associations between group attributes and scores are descriptive and do not establish causation.
- The dataset is limited in size; a single train-test split does not establish generalization to other populations.
- Do not use this demonstration to make admissions, placement, disciplinary, or other high-stakes decisions about students.

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Include residual analysis and prediction intervals
- Compare alternative feature sets, including a model without reading and writing scores
- Add a model card describing intended use and limitations
- Deploy the dashboard to Streamlit Community Cloud

## Author

**Nadeem Ahamad**


Data Science Internship Project associated with **Hackveda Solutions Private Limited Internship**, focused on student performance analysis, predictive modeling, model evaluation, and interactive visualization using Python and Scikit-learn.

