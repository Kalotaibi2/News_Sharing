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
    try:
        # If input_data is a dictionary (manual input), convert it to DataFrame; otherwise, assume it's already a DataFrame
        if isinstance(input_data, dict):
            input_data_df = pd.DataFrame([input_data])  # Convert dictionary to DataFrame (manual input)
        else:
            input_data_df = input_data  # Already a DataFrame (CSV file)
        
        # Ensure column names are clean
        input_data_df.columns = input_data_df.columns.str.strip()  # Strip any leading/trailing spaces
        
        # Define the expected columns (after removing "Average Keyword Performance")
        expected_columns = ["n_tokens_title", "n_tokens_content", "num_hrefs", "num_imgs"]
        
        # Check if any of the expected columns are missing
        missing_columns = [col for col in expected_columns if col not in input_data_df.columns]
        
        if missing_columns:
            st.error(f"Uploaded file is missing columns: {', '.join(missing_columns)}")
            return None  # If there are missing columns, stop further processing

        # Ensure the correct column order
        input_data_df = input_data_df[expected_columns]
        
        # Apply scaling, interaction terms, and dimensionality reduction (from trained models)
        data_scaled = scaler.transform(input_data_df)
        data_poly = poly.transform(data_scaled)
        data_reduced = fld.transform(data_poly)
        
        return data_reduced
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Initialize processed_data to prevent undefined errors
processed_data = None

# Option 1: Manual Input
if input_method == "Manual Input":
    st.write("Enter values for key features:")
    
    n_tokens_title = st.number_input('Number of Words in Title', min_value=1)
    n_tokens_content = st.number_input('Number of Words in Content', min_value=1)
    num_hrefs = st.number_input('Number of Hyperlinks', min_value=0)
    num_imgs = st.number_input('Number of Images', min_value=0)

    # Collect user inputs in Streamlit
    input_data = {
        "n_tokens_title": n_tokens_title,
        "n_tokens_content": n_tokens_content,
        "num_hrefs": num_hrefs,
        "num_imgs": num_imgs,
    }

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
if processed_data is not None:
    if model_choice == "Logistic Regression":
        predictions = log_reg_model.predict(processed_data)
    elif model_choice == "LDA":
        predictions = lda_model.predict(processed_data)
    else:
        predictions = nn_model.predict(processed_data)

    # Map numerical predictions to readable categories
    category_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_categories = [category_map[pred] for pred in predictions]

    # If the predictions list contains only one element, display it directly
    if len(predicted_categories) == 1:
        st.write(f"Predicted Share Category: {predicted_categories[0]}")
    else:
        st.write("Predicted Share Categories (Low, Medium, High):")
        st.write(predicted_categories)

else:
    st.warning("Please upload a valid file or input correct data.")

# Optional: Display training metrics for reference
st.write("Note: Validation results from the training phase are:")
st.write("Logistic Regression: Accuracy: 80%, Precision: 78%, Recall: 79%, F1-Score: 78%")
st.write("LDA: Accuracy: 79%, Precision: 77%, Recall: 78%, F1-Score: 77%")
st.write("Neural Network: Accuracy: 81%, Precision: 80%, Recall: 80%, F1-Score: 80%")
