
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Healthcare Analytics", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Analytics & High-Risk Patient Prediction")
st.caption("Educational analytics demonstration • Not for clinical decision-making")

DATA_PATH = Path(__file__).parent / "Healthcare_Cleaned.csv"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_healthcare_model.joblib"
META_PATH = ARTIFACT_DIR / "model_metadata.joblib"
RESULTS_PATH = ARTIFACT_DIR / "model_comparison.csv"

@st.cache_data
def load_data():
    d = pd.read_csv(DATA_PATH)
    d.columns = [re.sub(r"_+", "_", re.sub(r"\s+", "_", str(c).strip())).strip("_") for c in d.columns]
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].astype("string").str.strip()
    for c in ["Admission_Date", "Discharge_Date"]:
        if c in d:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    return d

def make_training_data(d):
    work = d[d["High_Risk_Flag"].isin(["High Risk", "Standard"])].copy()
    y = work["High_Risk_Flag"].map({"Standard": 0, "High Risk": 1}).astype(int)
    if "Admission_Date" in work:
        work["Admission_Date"] = pd.to_datetime(work["Admission_Date"], errors="coerce")
        work["Admission_Month_Feature"] = work["Admission_Date"].dt.month
        work["Admission_Weekday_Feature"] = work["Admission_Date"].dt.dayofweek
        work["Admission_Year_Feature"] = work["Admission_Date"].dt.year
    excluded = {
        "High_Risk_Flag", "Name", "Patient_Name", "Doctor", "Hospital",
        "Admission_Date", "Discharge_Date", "Age_Group", "Billing_Category",
        "Admission_Year", "Admission_Month", "Billing_vs_Avg", "LOS_Category"
    }
    X = work.drop(columns=[c for c in excluded if c in work], errors="ignore")
    X = X.dropna(axis=1, how="all")
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    return X, y, prep, cat, num

@st.cache_resource
def get_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(META_PATH), None
    d = load_data()
    X, y, prep, cat, num = make_training_data(d)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
                                 colsample_bytree=0.85, eval_metric="logloss", scale_pos_weight=pos_weight,
                                 random_state=42, n_jobs=-1)
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        p, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        rows.append({"Model":name, "Accuracy":accuracy_score(yte,p),
                     "Precision":precision_score(yte,p,zero_division=0),
                     "Recall":recall_score(yte,p,zero_division=0),
                     "F1":f1_score(yte,p,zero_division=0),
                     "ROC_AUC":roc_auc_score(yte,proba)})
        fitted[name] = pipe
    comparison = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    meta = {"model_name":best_name, "feature_columns":X.columns.tolist(),
            "categorical_features":cat, "numeric_features":num}
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    joblib.dump(meta, META_PATH)
    comparison.to_csv(RESULTS_PATH, index=False)
    return fitted[best_name], meta, comparison

df = load_data()
model, metadata, trained_comparison = get_model()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Exploratory Analysis", "Model Performance", "Predict a Patient"])
    st.divider()
    st.caption("Dataset: Healthcare_Cleaned.csv")

if page == "Overview":
    st.subheader("Dataset overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Patient records", f"{len(df):,}")
    c2.metric("Features", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.warning("The provided CSV contains High_Risk_Flag, not readmitted_30_days. This app predicts the existing High Risk / Standard label only.")
    if "High_Risk_Flag" in df:
        counts = df["High_Risk_Flag"].value_counts().rename_axis("Risk Flag").reset_index(name="Patients")
        st.plotly_chart(px.pie(counts, names="Risk Flag", values="Patients", title="Risk Flag Distribution"), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.subheader("Column types and missing values")
    summary = pd.DataFrame({"Data type":df.dtypes.astype(str), "Missing":df.isna().sum(),
                            "Unique values":df.nunique(dropna=False)})
    st.dataframe(summary, use_container_width=True)

elif page == "Exploratory Analysis":
    st.subheader("Interactive exploratory data analysis")
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object","string","category"]).columns.tolist()
    a,b = st.columns(2)
    with a:
        col = st.selectbox("Numeric variable", numeric, index=numeric.index("Age") if "Age" in numeric else 0)
        st.plotly_chart(px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"), use_container_width=True)
    with b:
        cat = st.selectbox("Category variable", categorical, index=categorical.index("Medical_Condition") if "Medical_Condition" in categorical else 0)
        vc = df[cat].value_counts().head(20).rename_axis(cat).reset_index(name="Count")
        st.plotly_chart(px.bar(vc, x=cat, y="Count", title=f"Top categories: {cat}"), use_container_width=True)
    if {"Medical_Condition","Billing_Amount"}.issubset(df.columns):
        agg = df.groupby("Medical_Condition", as_index=False)["Billing_Amount"].mean()
        st.plotly_chart(px.bar(agg, x="Medical_Condition", y="Billing_Amount", title="Average billing by medical condition"), use_container_width=True)
    if {"High_Risk_Flag","LOS_Days"}.issubset(df.columns):
        st.plotly_chart(px.box(df, x="High_Risk_Flag", y="LOS_Days", title="Length of stay by risk flag"), use_container_width=True)
    if len(numeric) > 1:
        st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=".2f", title="Numeric correlation heatmap", aspect="auto"), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Four-model comparison")
    if trained_comparison is not None:
        comp = trained_comparison
    elif RESULTS_PATH.exists():
        comp = pd.read_csv(RESULTS_PATH)
    else:
        st.info("Model metrics will be created during model training.")
        comp = None
    if comp is not None:
        st.dataframe(comp.style.format({c:"{:.3f}" for c in comp.columns if c != "Model"}), use_container_width=True)
        long = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                               title="Accuracy, precision, recall, F1 and ROC-AUC"), use_container_width=True)
        st.success(f"Selected model by ROC-AUC: {metadata['model_name']}")
    else:
        st.write("Selected model:", metadata["model_name"])

elif page == "Predict a Patient":
    st.subheader("Enter admission details")
    st.info("The output estimates the supplied dataset's High_Risk_Flag category. It is not a readmission probability or medical diagnosis.")
    X_template, _, _, cats, nums = make_training_data(df)
    with st.form("patient_form"):
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(metadata["feature_columns"]):
            with cols[i % 2]:
                if feature in metadata["categorical_features"]:
                    options = sorted(df[feature].dropna().astype(str).unique().tolist())
                    if not options:
                        options = ["Unknown"]
                    values[feature] = st.selectbox(feature.replace("_"," "), options)
                else:
                    series = pd.to_numeric(df[feature], errors="coerce").dropna()
                    default = float(series.median()) if len(series) else 0.0
                    minv = float(series.min()) if len(series) else 0.0
                    maxv = float(series.max()) if len(series) else max(default, 1.0)
                    if minv == maxv:
                        values[feature] = st.number_input(feature.replace("_"," "), value=default)
                    else:
                        values[feature] = st.number_input(feature.replace("_"," "), min_value=minv, max_value=maxv, value=min(max(default,maxv),maxv))
        submitted = st.form_submit_button("Predict risk category", type="primary")
    if submitted:
        input_df = pd.DataFrame([values], columns=metadata["feature_columns"])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0,1])
        result = "High Risk" if prediction == 1 else "Standard"
        if result == "High Risk":
            st.error(f"Model classification: {result}")
        else:
            st.success(f"Model classification: {result}")
        st.metric("Model-estimated High Risk score", f"{probability:.1%}")
        st.progress(probability)
        st.caption("This score is model output for the dataset label, not a calibrated clinical risk estimate.")
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Project scope: healthcare admission analytics and High_Risk_Flag classification. Validate data definitions, model performance and fairness before any operational use.")
