import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Water Potability App", layout="wide")

st.title("💧 Water Potability Prediction System")

# Load model
model = joblib.load("water_model.pkl")

# Upload CSV
file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    if "Potability" in df.columns:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]

        y_pred = model.predict(X)
        df["Predicted_Potability"] = y_pred

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        st.subheader("📈 Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{report['1']['precision']:.2f}")
        col3.metric("Recall", f"{report['1']['recall']:.2f}")
        col4.metric("F1 Score", f"{report['1']['f1-score']:.2f}")

    else:
        y_pred = model.predict(df)
        df["Predicted_Potability"] = y_pred
        st.warning("⚠️ No actual values → Only predictions shown")

    # 🔥 COUNT DISPLAY
    st.subheader("📊 Prediction Summary")

    safe = (df["Predicted_Potability"] == 1).sum()
    unsafe = (df["Predicted_Potability"] == 0).sum()

    col1, col2 = st.columns(2)
    col1.success(f"✅ Safe Water: {safe}")
    col2.error(f"❌ Unsafe Water: {unsafe}")

    # 🔥 GRAPH
    st.subheader("📉 Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Safe", "Unsafe"], [safe, unsafe])
    ax.set_ylabel("Count")
    ax.set_title("Water Potability Prediction")

    st.pyplot(fig)

    # 🔥 OUTPUT TABLE
    st.subheader("📄 Final Output")
    st.dataframe(df)

    # 🔥 DOWNLOAD BUTTON
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="water_predictions.csv",
        mime="text/csv"
    )