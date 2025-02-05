import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model class
class FCNN_Model(nn.Module):
    def __init__(self):
        super(FCNN_Model, self).__init__()
        
        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(42, 1024)  # 42 input features
        self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% rate
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 22)  # Output layer (for multi-class classification)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

# Initialize the model (important for both saving and loading)
model = FCNN_Model()

# Load the saved model
try:
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = FCNN_Model()  # If there's an error, use an empty model

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
        transition: transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF5722;
        transform: scale(1.1);
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
        transition: transform 0.3s ease;
    }
    .stSidebar .stButton>button:hover {
        background-color: #FF5722;
        transform: scale(1.1);
    }
    .stSidebar .stNumberInput>div>div>input {
        border: 2px solid #FF7043;
        border-radius: 8px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FF7043;
        color: white;
        text-align: center;
        padding: 10px;
        animation: slideUp 1s ease-out;
    }
    @keyframes slideUp {
        from {
            transform: translateY(100%);
        }
        to {
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# NSL-KDD feature names
feature_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Data Visualization", "About"])

if page == "Home":
    st.title("🌐 Network Intrusion Detection System")
    st.write("""
    Welcome to the Network Intrusion Detection System. This application uses a machine learning model to predict whether a network activity is normal or an intrusion.
    Use the navigation menu to explore different sections of the app.
    """)
    st.image("network_security.jpg", use_container_width=True)

elif page == "Predict":
    st.title("🔍 Predict Network Intrusion")
    st.write("Enter data for prediction")

    # Create input fields for features
    input_data = []
    for feature in feature_names:
        input_data.append(st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.01))

    if st.button('Predict'):
        prediction = predict([input_data])
        st.write(f"The predicted class is: **{prediction}**")

elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    st.write("Explore the dataset and visualize different features")

    # File uploader for CSV
    # File uploader to choose CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
    # Load the dataset
        df = pd.read_csv(uploaded_file)
    
        # Display the dataset
        st.write("### Dataset")
        st.write(df.head())

    # Handle missing values (fill NaNs with 0 or another strategy)
        df = df.fillna(0)

    # Select numeric columns for correlation and visualization
        df_numeric = df.select_dtypes(include=[np.number])

    # Feature distribution plot
        st.write("### Feature Distribution")
        feature = st.selectbox("Select a feature to visualize", df_numeric.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(df_numeric[feature], kde=True, color='#FF7043')
        plt.title(f'Distribution of {feature}')
        st.pyplot(plt)

    # Correlation matrix
        st.write("### Correlation Matrix")
        corr_matrix = df_numeric.corr()
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
