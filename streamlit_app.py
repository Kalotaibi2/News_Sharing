import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pre-trained models and preprocessing objects
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_model.pkl')
fld = joblib.load('fld_model.pkl')

log_reg_model = joblib.load('Logistic Regression_model.pkl')
lda_model = joblib.load('LDA_model.pkl')
nn_model = joblib.load('Neural Network_model.pkl')

# Streamlit App
st.title("News Sharing Prediction App")

# Provide an option to input data manually or upload a CSV file
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Function to preprocess data (manual input or from CSV)
def preprocess_data(input_data):
    # Convert input to DataFrame and strip spaces in column names
    input_data_df = pd.DataFrame([input_data])
    input_data_df.columns = input_data_df.columns.str.strip()  # Ensure no leading/trailing spaces
    input_data_df = input_data_df[scaler.feature_names_in_]  # Ensure same column order and names as training

    # Apply scaling, interaction terms, and dimensionality reduction
    data_scaled = scaler.transform(input_data_df)
    data_poly = poly.transform(data_scaled)
    data_reduced = fld.transform(data_poly)
    return data_reduced

# Option 1: Manual Input
if input_method == "Manual Input":
    st.write("Enter values for key features:")
    
    n_tokens_title = st.number_input('Number of Words in Title', min_value=1)
    n_tokens_content = st.number_input('Number of Words in Content', min_value=1)
    num_hrefs = st.number_input('Number of Hyperlinks', min_value=0)
    num_imgs = st.number_input('Number of Images', min_value=0)
    kw_avg_avg = st.number_input('Average Keyword Performance', min_value=0.0)

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'n_tokens_title': [n_tokens_title],
        'n_tokens_content': [n_tokens_content],
        'num_hrefs': [num_hrefs],
        'num_imgs': [num_imgs],
        'kw_avg_avg': [kw_avg_avg]
    })

    # Preprocess the input data
    processed_data = preprocess_data(input_data)

# Option 2: Upload CSV
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(input_data.head())
        
        # Preprocess the uploaded data
        processed_data = preprocess_data(input_data)

# Model Selection
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Logistic Regression", "LDA", "Neural Network"])

# Make predictions based on model choice
if model_choice == "Logistic Regression":
    predictions = log_reg_model.predict(processed_data)
elif model_choice == "LDA":
    predictions = lda_model.predict(processed_data)
else:
    predictions = nn_model.predict(processed_data)

# Display results
st.write("Predicted Share Categories (Low, Medium, High):")
st.write(predictions)

# Since this is a live demo, accuracy metrics can't be calculated unless there is labeled data
st.write("Note: Since this is a live demo, accuracy metrics are not calculated without true labels.")
