import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model architecture first
model = FCNN_Model()  # Use your specific model class here

# Load the state_dict into the model
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()


# Define your prediction function
def predict(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #FFF3E0;
        color: #5D4037;
    }
    .stButton>button {
        background-color: #FF7043;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #FF5722;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #FF7043;
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background-color: #FFE0B2;
    }
    .stSidebar .stButton>button {
        background-color: #FF7043;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stSidebar .stButton>button:hover {
        background-color: #FF5722;
    }
    .stSidebar .stNumberInput>div>div>input {
        border: 2px solid #FF7043;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Data Visualization", "About"])

if page == "Home":
    st.title("🌐 Network Intrusion Detection System")
    st.write("""
    Welcome to the Network Intrusion Detection System. This application uses a machine learning model to predict whether a network activity is normal or an intrusion.
    Use the navigation menu to explore different sections of the app.
    """)
    st.image("network_security.jpg", use_column_width=True)

elif page == "Predict":
    st.title("🔍 Predict Network Intrusion")
    st.write("Enter data for prediction")

    # Create input fields for features (for example, 42 features)
    input_data = []
    for i in range(42):  # 42 features
        input_data.append(st.number_input(f"Feature {i+1}", min_value=0.0, max_value=100.0, step=0.01))

    if st.button('Predict'):
        prediction = predict([input_data])
        st.write(f"The predicted class is: **{prediction}**")

elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    st.write("Explore the dataset and visualize different features")

    # Load a sample dataset
    df = pd.read_csv('network_data.csv')

    st.write("### Dataset")
    st.write(df.head())

    st.write("### Feature Distribution")
    feature = st.selectbox("Select a feature to visualize", df.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color='#FF7043')
    plt.title(f'Distribution of {feature}')
    st.pyplot(plt)

    st.write("### Correlation Matrix")
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

elif page == "About":
    st.title("ℹ️ About")
    st.write("""
    This application is developed to help in detecting network intrusions using machine learning.
    The model is trained on a dataset of network activities and can predict whether an activity is normal or an intrusion.
    """)

    st.write("### Technologies Used")
    st.write("""
    - **Streamlit**: For building the web application
    - **PyTorch**: For building and training the machine learning model
    - **Pandas**: For data manipulation and analysis
    - **Matplotlib & Seaborn**: For data visualization
    """)

    st.write("### Developer")
    st.write("Developed by [Your Name].")

    st.write("### Contact")
    st.write("For any queries, please contact [your.email@example.com].")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FF7043;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>© 2023 Network Intrusion Detection System. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
