import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Water Potability Dashboard",
    page_icon="💧",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("water_model.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("💧 Water Quality App")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Batch Prediction", "🧪 Single Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("Built by Varun | Data Analyst")

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("💧 Water Potability Prediction Dashboard")

    st.markdown("""
    ### 🚀 Project Overview
    This dashboard predicts whether water is **safe (potable)** or **unsafe**  
    using a Machine Learning model.

    ### 🔍 Features
    - Batch CSV prediction
    - Real-time prediction
    - Interactive charts
    - Download results
    """)

# ---------------- BATCH PREDICTION ----------------
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Prediction (Upload CSV)")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # ---------------- PREDICTION ----------------
        if "Potability" in df.columns:
            X = df.drop("Potability", axis=1)
        else:
            X = df.copy()

        df["Prediction"] = model.predict(X)

        # ---------------- KEY METRICS ----------------
        st.subheader("📌 Key Metrics")

        safe = (df["Prediction"] == 1).sum()
        unsafe = (df["Prediction"] == 0).sum()
        total = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", total)
        col2.metric("Safe Water ✅", safe)
        col3.metric("Unsafe Water ❌", unsafe)

        # ---------------- MODEL PERFORMANCE ----------------
        if "Potability" in df.columns:
            y_true = df["Potability"]
            y_pred = df["Prediction"]

            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            st.subheader("📈 Model Performance")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{report['1']['precision']:.2f}")
            col3.metric("Recall", f"{report['1']['recall']:.2f}")
            col4.metric("F1 Score", f"{report['1']['f1-score']:.2f}")
        else:
            st.info("ℹ️ Upload dataset with 'Potability' column to view model performance.")

        # ---------------- CHARTS ----------------
        st.subheader("📊 Visualization")

        pie_fig = px.pie(
            names=["Safe", "Unsafe"],
            values=[safe, unsafe],
            title="Water Potability Distribution"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        bar_fig = px.bar(
            x=["Safe", "Unsafe"],
            y=[safe, unsafe],
            title="Water Quality Comparison"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # ---------------- OUTPUT ----------------
        st.subheader("📄 Final Output")
        st.dataframe(df)

        # ---------------- DOWNLOAD ----------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Predictions",
            csv,
            "water_predictions.csv",
            "text/csv"
        )

# ---------------- SINGLE PREDICTION ----------------
elif page == "🧪 Single Prediction":
    st.title("🧪 Predict Water Quality")

    st.markdown("Enter water parameters:")

    col1, col2, col3 = st.columns(3)

    ph = col1.number_input("pH", 0.0, 14.0, 7.0)
    hardness = col2.number_input("Hardness", 0.0, 500.0, 200.0)
    solids = col3.number_input("Solids", 0.0, 50000.0, 15000.0)

    col4, col5, col6 = st.columns(3)

    chloramines = col4.number_input("Chloramines", 0.0, 20.0, 7.0)
    sulfate = col5.number_input("Sulfate", 0.0, 500.0, 300.0)
    conductivity = col6.number_input("Conductivity", 0.0, 1000.0, 400.0)

    col7, col8, col9 = st.columns(3)

    organic_carbon = col7.number_input("Organic Carbon", 0.0, 30.0, 10.0)
    trihalomethanes = col8.number_input("Trihalomethanes", 0.0, 200.0, 80.0)
    turbidity = col9.number_input("Turbidity", 0.0, 10.0, 4.0)

    if st.button("🔍 Predict"):
        input_data = pd.DataFrame([[
            ph, hardness, solids, chloramines,
            sulfate, conductivity, organic_carbon,
            trihalomethanes, turbidity
        ]], columns=[
            'ph', 'Hardness', 'Solids', 'Chloramines',
            'Sulfate', 'Conductivity', 'Organic_carbon',
            'Trihalomethanes', 'Turbidity'
        ])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Water is SAFE to drink")
        else:
            st.error("❌ Water is NOT SAFE to drink")

# ---------------- FOOTER ----------------
st.markdown("""
---
<center>💧 Water Potability ML Dashboard | Built with Streamlit</center>
""", unsafe_allow_html=True)