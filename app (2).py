import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
# Load dataset
def load_data():
    return pd.read_csv("Merged_Food_Hampers_and_Clients.csv")

data = load_data()

# Load models
my_model = joblib.load('best_model.pkl')

# Define pages for the Streamlit app
transaction_data = data.copy()

# Initialize the model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Limit to latest 5 transactions (or 10)
latest_transactions = transaction_data.tail(5)

transaction_narrative = "Here are the latest client transactions:\n"
for idx, row in latest_transactions.iterrows():
    if pd.isna(row['pickup_date']) or pd.isna(row['primary_client_key']):
        continue  # skip incomplete rows

    transaction_narrative += (
        f"Client {row['unique_client']} picked up a {row['hamper_type']} hamper "
        f"at {row['pickup_location']} on {row['pickup_date'].strftime('%Y-%m-%d')}.\n"
    )


# Charity Info
charity_info = (
    "IFSSA is a non-profit helping families in Edmonton. It offers food hampers, family support, "
    "housing help, and mental health services."
)


# System message to control chatbot behavior
system_message = """
You are a helpful assistant for IFSSA.
Answer ONLY questions related to IFSSA, food hampers, recent transactions, or the charity's services.
Avoid answering unrelated questions or generating irrelevant information.
"""

# Now, define the documents dictionary with the narrative and charity info
documents = {
    "doc1": charity_info,
    "doc2": transaction_narrative
}

# Generate embeddings for each document
doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

# Initialize the model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def query_llm(prompt, context):
    """
    Function to query the LLM with the given prompt and context.

    Args:
    prompt (str): The question or query to ask the model.
    context (str): The context or background information for the model to consider.

    Returns:
    str: The model's response.
    """
    # Combine context and prompt into a full input
    full_input = context + "\n" + prompt

    # Tokenize the input, ensuring it does not exceed the model's token limit
    inputs = tokenizer(full_input, return_tensors='pt', truncation=True, max_length=1024)

    # Debugging: print the shape of the tokenized input (for debugging purposes)
    print(inputs['input_ids'].shape)  # Output the shape of input_ids to check its length

    # Calculate remaining tokens (assuming model max length is 1024)
    input_length = inputs['input_ids'].shape[1]
    max_new_tokens = 1024 - input_length  # Allow space for new tokens

    # Generate a response from the model, ensuring it stays within the token limit
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,  # Limit the number of new tokens generated
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Temperature for randomness (higher = more random)
        num_return_sequences=1  # Return only one sequence
    )

    # Decode the output tokens back into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def retrieve_context(query, top_k=1):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {}
    for doc_id, emb in doc_embeddings.items():
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        scores[doc_id] = score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
    context = "\n\n".join(documents[doc_id] for doc_id in top_doc_ids)
    return context

def rag_chatbot(query):
    context = retrieve_context(query, top_k=1)
    answer = query_llm(query, context)
    return answer

# Chatbot Page
def chatbot_page():
    st.title("IFSSA Virtual Assistant")

    

    # Function to generate response
    def generate_response(user_input):
      # Simple greeting check
      greetings = ['hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!']
      if any(greeting in user_input.lower() for greeting in greetings):
          return "Hi! How can I help you today?"

      # If it's not a greeting, you can process the request based on context here.
      # For example, if it's a query about services or transactions, you'd proceed with that.
      # For now, we return a placeholder response.
      return "Please ask me about IFSSA services or recent transactions."

      user_input = st.text_input("Ask me anything:")

      # Show response based on user input
      if user_input:
        response = generate_response(user_input)
        st.write(response)

    
# Dashboard Page with Problem Statement and Image
def dashboard():
    st.title("📊 Dashboard Overview")
    st.image("foodhamperimage.png", caption="Food Hamper Delivery Process", use_container_width=True)
    st.write("""
    **Problem Statement:**
    The goal of this project is to predict delivery delays for food hampers based on various features like delivery hour, communication barriers, distance, and more. By leveraging predictive modeling, we aim to optimize the delivery process and ensure timely delivery, reducing unnecessary delays.
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
    numeric_data = data.select_dtypes(include=np.number)
    fig = px.imshow(numeric_data.corr(), text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(fig)
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
        prediction = my_model.predict(input_df_scaled)[0]
        proba = my_model.predict_proba(input_df_scaled)[0]

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

# Chatbot Page
def chatbot_page():
    st.title("💬 Chat with the RAG Chatbot")
    user_query = st.text_input("Ask a question about the data:")

    if user_query:
        response = rag_chatbot(user_query)
        st.write("Answer:", response)

# Thank You Page
def thank_you_page():
    st.title("🙏 Thank You!")
    st.write("""
    Thank you for exploring the **Food Hamper Delivery Prediction** app! We hope you found it useful in understanding the predictive modeling process for delivery delays. If you have any questions or feedback, feel free to reach out.
    """)

# Main App Logic
def main():
    st.sidebar.title("Food Hamper Delivery Prediction")
    page = st.sidebar.radio("Select a Page", ["Dashboard", "Dataset Overview", "EDA", "Prediction", "Chatbot", "Thank You"])

    if page == "Dashboard":
        dashboard()
    elif page == "Dataset Overview":
        dataset_overview()
    elif page == "EDA":
        exploratory_data_analysis()
    elif page == "Prediction":
        predict_page()
    elif page == "Chatbot":
        chatbot_page()
    elif page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
