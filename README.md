# Water-Potability-Prediction
Water Potability Prediction
📌 Project Overview

This project focuses on predicting whether water is safe for drinking (potable) using machine learning techniques. By analyzing various water quality parameters such as pH, hardness, solids, chloramines, and others, the model determines the potability of water.

The goal is to assist in public health decision-making by providing a data-driven approach to identify safe and unsafe water sources.

🎯 Objectives
Analyze water quality data
Perform data cleaning and preprocessing
Build a machine learning model for classification
Predict whether water is potable or not
Deploy a web application for real-time predictions
📊 Dataset Features

The dataset contains the following attributes:

pH – Acidic or basic nature of water
Hardness – Concentration of minerals
Solids – Total dissolved solids
Chloramines – Disinfectants in water
Sulfate – Sulfate concentration
Conductivity – Electrical conductivity
Organic Carbon – Organic matter in water
Trihalomethanes – Chemical compounds in water
Turbidity – Clarity of water
Potability – Target variable (0 = Not Safe, 1 = Safe)

🛠️ Technologies Used
Python
Pandas, NumPy (Data Processing)
Matplotlib, Seaborn (Visualization)
Scikit-learn (Machine Learning)
Streamlit (Deployment)

⚙️ Project Workflow
1. Data Preprocessing
Handled missing values
Performed feature scaling
Cleaned dataset for better accuracy

3. Exploratory Data Analysis (EDA)
Visualized relationships between features
Identified important factors affecting water quality

5. Model Building
Used classification algorithms (e.g., Random Forest / Logistic Regression)
Trained model on processed dataset

7. Model Evaluation
Achieved approximately 68% accuracy
Evaluated using:
Precision
Recall
F1-score

8. Deployment
Built a Streamlit web app
Users can upload CSV or input values to get predictions

🚀 How to Run the Project
🔹 Step 1: Clone the Repository
git clone https://github.com/your-username/water-potability-prediction.git
cd water-potability-prediction

🔹 Step 2: Install Dependencies
pip install -r requirements.txt

🔹 Step 3: Run the Application
streamlit run app.py

📸 Output
Predicts whether water is Safe (1) or Not Safe (0)
Displays results instantly in the web interface

📈 Future Improvements
Improve model accuracy using advanced algorithms
Add real-time IoT data integration
Enhance UI/UX of the application
Deploy on cloud (AWS / Render / Streamlit Cloud)

🙌 Conclusion

This project demonstrates how machine learning can be used to solve real-world problems like water safety. It provides a practical approach to analyzing environmental data and making informed decisions.
