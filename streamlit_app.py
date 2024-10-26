import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load pre-trained models and preprocessing objects
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_model.pkl')
decision_tree_model = joblib.load('Decision_Tree_model.pkl')
naive_bayes_model = joblib.load('Naive_Bayes_model.pkl')
random_forest_model = joblib.load('Random_Forest_model.pkl')

# Precomputed average self_reference_avg_sharess from the training set
avg_self_reference_shares = 3395
default_self_reference_min_shares = 200
default_kw_avg_avg = 0.5
default_kw_max_avg = 0.8
default_data_channel_is_world = 0
default_LDA_02 = 0.2
default_LDA_03 = 0.3

# Streamlit App
st.title("News Sharing Prediction App")
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV", "View Preprocessing Results"])

# Function to preprocess data with feature consistency check
def preprocess_data(input_data):
    expected_features = [
        'kw_avg_avg', 'LDA_03', 'kw_max_avg', 'self_reference_avg_sharess',
        'self_reference_min_shares', 'data_channel_is_world', 'LDA_02',
        'num_hrefs', 'num_imgs'
    ]
    if isinstance(input_data, dict):
        input_data_df = pd.DataFrame([input_data])
    else:
        input_data_df = input_data

    input_data_df.columns = input_data_df.columns.str.strip()
    for feature in expected_features:
        if feature not in input_data_df.columns:
            input_data_df[feature] = 0  # Fill missing columns with default values

    input_data_df = input_data_df[expected_features]
    data_scaled = scaler.transform(input_data_df)
    data_poly = poly.transform(data_scaled)
    return data_poly

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
        'self_reference_avg_sharess': avg_self_reference_shares,
        'self_reference_min_shares': default_self_reference_min_shares,
        'kw_avg_avg': default_kw_avg_avg,
        'kw_max_avg': default_kw_max_avg,
        'data_channel_is_world': default_data_channel_is_world,
        'LDA_02': default_LDA_02,
        'LDA_03': default_LDA_03,
    }
    
    processed_data = preprocess_data(input_data)

# Option 2: Upload CSV
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(input_data.head())
        processed_data = preprocess_data(input_data)

# Option 3: View Preprocessing Results
elif input_method == "View Preprocessing Results":
    st.subheader("Preprocessing Steps and Visualizations")

    # Overview Button: Display first 5 rows and a brief description
    if st.button("Overview"):
        data = pd.read_csv("OnlineNewsPopularity.csv")
        st.write("Dataset Overview:")
        st.write(data.head())
        st.write("This dataset includes various features about news articles and their popularity in terms of shares.")

    # Correlation Button: Display correlation heatmap
    if st.button("Correlation"):
        st.write("Correlation Matrix of Selected Features")
        image = Image.open('correlation_matrix.png')
        st.image(image, caption='Correlation Matrix', use_column_width=True)

    # Threshold Button: Display thresholds for categorization
    if st.button("Threshold"):
        with open('categorize_thresholds.txt', 'r') as file:
            thresholds = file.read()
        st.write("Thresholds for Categorization:")
        st.text(thresholds)

    # Outlier Button: Display boxplot for outliers
    if st.button("Outliers"):
        st.write("Outlier Visualization")
        outlier_image = Image.open('outlier_visualization.png')
        st.image(outlier_image, caption='Boxplot of Key Features to Identify Outliers', use_column_width=True)

    # Evolution Button: Comparison of models' performance
    if st.button("Model Evolution"):
        st.write("Model Performance Comparison")
        results_df = pd.read_csv('model_evaluation_results.csv')
        st.write(results_df)

# Model Selection for prediction
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Decision Tree", "Naive Bayes", "Random Forest"])

# Make predictions based on the selected model
if 'processed_data' in locals():
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

# Note
st.write("Note: These results are based on the trained models and precomputed analysis. Live predictions will vary with new data.")
