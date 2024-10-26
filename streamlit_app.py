import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained models and preprocessing objects
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_model.pkl')
fld = joblib.load('fld_model.pkl')
decision_tree_model = joblib.load('Decision Tree_model.pkl')
naive_bayes_model = joblib.load('Naive Bayes_model.pkl')
random_forest_model = joblib.load('Random Forest_model.pkl')

# Streamlit App
st.title("News Sharing Prediction App")
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Function to preprocess data
def preprocess_data(input_data):
    # Convert input to DataFrame if manual input
    if isinstance(input_data, dict):
        input_data_df = pd.DataFrame([input_data])
    else:
        input_data_df = input_data

    input_data_df.columns = input_data_df.columns.str.strip()

    # Define the expected columns for derived features computation
    expected_columns = ['n_tokens_content', 'num_hrefs', 'num_imgs', 'self_reference_avg_sharess']

    # Ensure correct columns and fill missing ones with default values
    for col in expected_columns:
        if col not in input_data_df.columns:
            input_data_df[col] = 0

    input_data_df = input_data_df[expected_columns]
    
    # Apply scaling, polynomial features, and dimensionality reduction
    data_scaled = scaler.transform(input_data_df)
    data_poly = poly.transform(data_scaled)
    data_reduced = fld.transform(data_poly)
    
    return data_reduced

#avg_self_reference_shares is calculated as the mean from the training set:
avg_self_reference_shares = training_data['self_reference_avg_sharess'].mean()

# Replace 'self_reference_avg_sharess' with this calculated value:
input_data['self_reference_avg_sharess'] = avg_self_reference_shares
# Option 1: Manual Input
if input_method == "Manual Input":
    st.write("Enter basic article features:")
    n_tokens_content = st.number_input('Number of Words in Content', min_value=0)
    num_hrefs = st.number_input('Number of Hyperlinks', min_value=0)
    num_imgs = st.number_input('Number of Images', min_value=0)
    

    input_data = {
        'n_tokens_content': n_tokens_content,
        'num_hrefs': num_hrefs,
        'num_imgs': num_imgs,
        'self_reference_avg_sharess': self_reference_avg_sharess,
    }
    
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
model_choice = st.sidebar.selectbox(
    "Select a model:", ["Decision Tree", "Naive Bayes", "Random Forest"])

# Make predictions based on model choice
if processed_data is not None:
    if model_choice == "Decision Tree":
        predictions = decision_tree_model.predict(processed_data)
    elif model_choice == "Naive Bayes":
        predictions = naive_bayes_model.predict(processed_data)
    elif model_choice == "Random Forest":
        predictions = random_forest_model.predict(processed_data)

    # Map numerical predictions to categories
    category_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_categories = [category_map[pred] for pred in predictions]

    if len(predicted_categories) == 1:
        st.write(f"Predicted Share Category: {predicted_categories[0]}")
    else:
        st.write("Predicted Share Categories:")
        st.write(predicted_categories)
else:
    st.warning("Please upload a valid file or input correct data.")

# Validation Note
st.write("Note: These results are based on the trained models. For validation, results may vary when using different data samples.")

# Visualize Outlier Handling for Documentation
st.subheader("Outlier Visualization")
outlier_features = ['n_tokens_content', 'num_hrefs', 'num_imgs', 'shares']
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=pd.read_csv("OnlineNewsPopularity.csv")[outlier_features], ax=ax)
ax.set_title("Boxplot of Key Features to Identify Outliers")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
