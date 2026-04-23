import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Diabetes Predictor+", layout="wide")
st.title("🧠 AI Diabetes Prediction System (Advanced Version)")
st.markdown("### With Explainable AI + Risk Intelligence + Recommendations")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

data = load_data()

# -------------------------------
# Data Cleaning (Novel Improvement)
# -------------------------------
# Replace zero values with median (important medical correction)
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_to_fix:
    data[col] = data[col].replace(0, data[col].median())

# -------------------------------
# Features / Labels
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Models
# -------------------------------
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Accuracy
lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# -------------------------------
# Sidebar Input
# -------------------------------
st.sidebar.header("📥 Patient Input")

preg = st.sidebar.slider("Pregnancies", 0, 15, 1)
glucose = st.sidebar.slider("Glucose", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 40, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 10, 100, 20)
insulin = st.sidebar.slider("Insulin", 15, 300, 80)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
age = st.sidebar.slider("Age", 18, 80, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.sidebar.button("🔍 Analyze Risk"):

    st.subheader("📊 Prediction Result")

    prob = lr.predict_proba(input_scaled)[0][1]
    pred = lr.predict(input_scaled)[0]

    # Output
    if pred == 1:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("✅ Low Diabetes Risk")

    st.write(f"### 🔢 Risk Score: {round(prob*100,2)}%")

    # Risk Category (Novel Logic)
    if prob < 0.3:
        level = "🟢 Low"
    elif prob < 0.7:
        level = "🟡 Moderate"
    else:
        level = "🔴 High"

    st.write(f"### 📌 Risk Level: {level}")

    # -------------------------------
    # AI Recommendation Engine
    # -------------------------------
    st.subheader("💡 AI Health Recommendations")

    recommendations = []

    if glucose > 140:
        recommendations.append("Reduce sugar intake and monitor glucose levels")
    if bmi > 30:
        recommendations.append("Start daily exercise and weight management")
    if age > 45:
        recommendations.append("Schedule regular diabetes screening")
    if insulin > 200:
        recommendations.append("Consult endocrinologist for insulin control")
    if bp > 90:
        recommendations.append("Maintain healthy blood pressure through diet")

    if recommendations:
        for r in recommendations:
            st.write(f"- {r}")
    else:
        st.write("✔️ Maintain your healthy lifestyle!")

    # -------------------------------
    # Explainable AI (SHAP)
    # -------------------------------
    st.subheader("🔍 Why this prediction? (Explainable AI)")

    explainer = shap.Explainer(lr, X_train)
    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# -------------------------------
# Model Comparison Section
# -------------------------------
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Logistic Regression", f"{lr_acc:.2f}")
col2.metric("Random Forest", f"{rf_acc:.2f}")

# -------------------------------
# Feature Insights (Novel)
# -------------------------------
st.subheader("🧬 Feature Importance Insight")

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

fig2, ax2 = plt.subplots()
importance.plot(kind="bar", ax=ax2)
st.pyplot(fig2)

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("🔥 Feature Correlation")

fig3, ax3 = plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)