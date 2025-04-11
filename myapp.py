import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
def load_data():
    return pd.read_csv("Merged_Food_Hampers_and_Clients.csv")

data = load_data()

# Load models
model = joblib.load('best_model.pkl')

# Dashboard Page with Problem Statement and Image
def dashboard():
    st.title("📊 Dashboard Overview")

    # Display an image in the dashboard (make sure the image is in the same directory or provide the URL)
    st.image("foodhamperimage.png", caption="Food Hamper Delivery Process", use_container_width=True)  # Update the filename accordingly

    st.write("""
    **Problem Statement:**
    The goal of this project is to predict delivery delays for food hampers based on various features like delivery hour, communication barriers, distance, and more. By leveraging predictive modeling, we aim to optimize the delivery process and ensure timely delivery, reducing unnecessary delays.

    The machine learning model used for predictions has been trained on a dataset of food hamper deliveries, with multiple features affecting the outcome. The app allows users to explore the data, run predictions, and visualize the results in an interactive manner.
    """)

# Dataset Overview Page
def dataset_overview():
    st.title("📊 Dataset Overview")
    st.write("Here is a closer look at the dataset used for this analysis.")

    st.subheader("Dataset Shape")
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

    st.subheader("Column Descriptions")
    st.write("""
    - **Delivery_Hour**: The hour of the day when the delivery is scheduled.
    - **communication_barrier**: Indicates if there is a communication barrier (0: No, 1: Yes).
    - **dependents_qty**: The number of dependents a person has.
    - **urgent_goal**: Whether the delivery is urgent (0: No, 1: Yes).
    - **distance_km**: The distance (in kilometers) for delivery.
    - **Delayed**: Target variable (0: On time, 1: Delayed).
    """)

    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Basic Statistics")
    st.write(data.describe())

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

# EDA Page (with additional visuals from your notebook)
def exploratory_data_analysis():
    st.title("📊 Exploratory Data Analysis")
    st.write("Basic statistics and visualizations.")

    st.subheader("Dataset Overview")
    st.write(data.head())

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature", data.columns)
    fig = px.histogram(data, x=selected_feature, title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=np.number)
    fig = px.imshow(numeric_data.corr(), text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(fig)

    # Add More Visualizations from your notebook
    st.subheader("Pairplot")
    fig = px.scatter_matrix(data)
    st.plotly_chart(fig)

# Prediction Page
def predict_page():
    st.title("🚚 Predict Delivery Delay")

    # Load the saved scaler
    scaler = joblib.load("scaler.pkl")

    # User Inputs
    delivery_hour = st.slider("Delivery Hour", 0, 23, 12)
    communication_barrier = st.selectbox("Communication Barrier", [0, 1])
    dependents_qty = st.slider("Number of Dependents", 0, 10, 2)
    urgent_goal = st.selectbox("Urgent Goal", [0, 1])
    organization_x = st.slider("Organization", min_value=0, max_value=1, step=1)

    if st.button("Predict Delay"):
        # Create input dataframe
        input_df = pd.DataFrame([[delivery_hour, communication_barrier, dependents_qty, urgent_goal, organization_x]],
                                columns=["Delivery_Hour", "communication_barrier", "dependents_qty", "urgent_goal", "organization_x"])

        # Ensure correct feature order (must match training data)
        expected_order = ["Delivery_Hour", "communication_barrier", "dependents_qty", "urgent_goal", "organization_x"]
        input_df = input_df[expected_order]

        # Apply the same scaling as training
        input_df_scaled = scaler.transform(input_df)

        # Predict using the trained model
        prediction = model.predict(input_df_scaled)[0]
        proba = model.predict_proba(input_df_scaled)[0]

        # Set threshold to balance predictions
        threshold = 0.55  # Adjust if necessary
        prediction = 1 if proba[1] > threshold else 0

        # Display Result
        result_text = "🚚 Delivery is Delayed" if prediction == 1 else "✅ Delivery is On Time"
        st.subheader(result_text)

        # Probability Visualization
        fig = go.Figure(go.Bar(x=[proba[0], proba[1]], y=["On Time", "Delayed"],
                               orientation='h', marker=dict(color=['green', 'red'])))
        fig.update_layout(title="Prediction Probabilities", xaxis_title="Probability", yaxis_title="Class")
        st.plotly_chart(fig)


# Thank You Page
def thank_you_page():
    st.title("🙏 Thank You!")
    st.write("""
    Thank you for exploring the **Food Hamper Delivery Prediction** app! We hope you found it useful in understanding the predictive modeling process for delivery delays. If you have any questions or feedback, feel free to reach out.

    Your feedback helps us improve the app and make it more effective. Stay tuned for more updates and enhancements!
    """)

# Main App Logic
def main():
    st.sidebar.title("Food Hamper Delivery Prediction")
    page = st.sidebar.radio("Select a Page", ["Dashboard", "Dataset Overview", "EDA", "Prediction", "Thank You"])

    if page == "Dashboard":
        dashboard()
    elif page == "Dataset Overview":
        dataset_overview()
    elif page == "EDA":
        exploratory_data_analysis()
    elif page == "Prediction":
        predict_page()
    elif page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
