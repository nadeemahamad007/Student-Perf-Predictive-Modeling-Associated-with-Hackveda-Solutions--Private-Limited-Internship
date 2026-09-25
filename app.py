from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Student Performance Explorer", page_icon="🎓", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "StudentsPerformance.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    required = {
        "Gender", "Race/Ethnicity", "Parental Level Of Education",
        "Lunch", "Test Preparation Course",
        "Math Score", "Reading Score", "Writing Score",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df

@st.cache_resource
def train_models(df):
    target = "Math Score"
    X, y = df.drop(columns=[target]), df[target]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    metrics, fitted = [], {}
    for name, est in estimators.items():
        pipe = Pipeline([("preprocess", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        metrics.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                        "R²": r2_score(y_test, pred)})
    return pd.DataFrame(metrics).sort_values("MAE"), fitted

if not DATA_PATH.exists():
    st.error("Dataset not found. Add data/StudentsPerformance.csv to the project folder.")
    st.stop()
df = load_data()
metrics, fitted = train_models(df)

st.title("🎓 Student Performance Explorer")
st.caption("Interactive analysis of assessment scores, student attributes, and baseline math-score models")
with st.sidebar:
    st.header("Explore the data")
    gender = st.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    prep = st.multiselect("Test preparation", sorted(df["Test Preparation Course"].unique()), default=sorted(df["Test Preparation Course"].unique()))
    lunch = st.multiselect("Lunch type", sorted(df["Lunch"].unique()), default=sorted(df["Lunch"].unique()))
    st.divider()
    st.caption("Filters affect the descriptive charts and summary metrics. Model evaluation uses a fixed 80/20 split of the full dataset.")
filtered = df[df["Gender"].isin(gender) & df["Test Preparation Course"].isin(prep) & df["Lunch"].isin(lunch)]
if filtered.empty:
    st.warning("No records match these filters. Select at least one option in each filter.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Students in view", f"{len(filtered):,}")
c2.metric("Average math", f"{filtered['Math Score'].mean():.1f}/100")
c3.metric("Average reading", f"{filtered['Reading Score'].mean():.1f}/100")
c4.metric("Average writing", f"{filtered['Writing Score'].mean():.1f}/100")

st.subheader("Score overview")
left,right=st.columns(2)
with left:
    long=filtered.melt(value_vars=["Math Score","Reading Score","Writing Score"],var_name="Assessment",value_name="Score")
    fig=px.histogram(long,x="Score",color="Assessment",barmode="overlay",opacity=.65,nbins=25,title="Score distributions")
    st.plotly_chart(fig,use_container_width=True)
with right:
    corr=filtered[["Math Score","Reading Score","Writing Score"]].corr().reset_index().melt(id_vars="index",var_name="Assessment",value_name="Correlation")
    fig=px.imshow(filtered[["Math Score","Reading Score","Writing Score"]].corr(),text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",title="Assessment score correlations")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Compare groups")
left,right=st.columns(2)
with left:
    fig=px.box(filtered,x="Test Preparation Course",y="Math Score",color="Test Preparation Course",points="outliers",title="Math score by test preparation")
    st.plotly_chart(fig,use_container_width=True)
with right:
    group=filtered.groupby("Parental Level Of Education",as_index=False)[["Math Score","Reading Score","Writing Score"]].mean().melt(id_vars="Parental Level Of Education",var_name="Assessment",value_name="Mean score")
    fig=px.bar(group,x="Parental Level Of Education",y="Mean score",color="Assessment",barmode="group",title="Average scores by parental education")
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Regression model evaluation")
st.caption("Target: Math Score. Predictors include the demographic fields plus Reading Score and Writing Score. Metrics are from a fixed held-out test set; they are not a guarantee of future performance.")
st.dataframe(metrics.round(3),use_container_width=True,hide_index=True)
fig=px.bar(metrics.melt(id_vars="Model",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error"),x="Model",y="Error",color="Metric",barmode="group",title="Held-out prediction error (lower is better)")
st.plotly_chart(fig,use_container_width=True)

st.subheader("Try a sample prediction")
st.write("Enter values to explore a model estimate. This is a demonstration, not an assessment of a real student's ability.")
with st.form("prediction_form"):
    a,b,c=st.columns(3)
    with a:
        p_gender=st.selectbox("Gender",sorted(df["Gender"].unique()))
        p_race=st.selectbox("Race/ethnicity group",sorted(df["Race/Ethnicity"].unique()))
    with b:
        p_parent=st.selectbox("Parental education",sorted(df["Parental Level Of Education"].unique()))
        p_lunch=st.selectbox("Lunch type",sorted(df["Lunch"].unique()))
    with c:
        p_prep=st.selectbox("Test preparation",sorted(df["Test Preparation Course"].unique()))
        p_read=st.slider("Reading score",0,100,70)
        p_write=st.slider("Writing score",0,100,70)
    model_options = metrics["Model"].tolist()
    model_name = st.selectbox("Model", model_options, index=0)
    submitted=st.form_submit_button("Estimate math score",type="primary")
if submitted:
    row=pd.DataFrame([{"Gender":p_gender,"Race/Ethnicity":p_race,"Parental Level Of Education":p_parent,"Lunch":p_lunch,"Test Preparation Course":p_prep,"Reading Score":p_read,"Writing Score":p_write}])
    estimate=float(fitted[model_name].predict(row)[0])
    st.success(f"Estimated Math Score: {estimate:.1f} / 100")
    st.caption(f"Model: {model_name}. The estimate is illustrative and should not be used for high-stakes decisions.")

with st.expander("View filtered records"):
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.download_button("Download filtered CSV",filtered.to_csv(index=False).encode("utf-8"),file_name="filtered_student_performance.csv",mime="text/csv")
st.caption("Source data: StudentsPerformance.csv. Group differences are descriptive and do not establish causation.")
