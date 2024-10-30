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
gradient_boosting_model = joblib.load('Gradient_Boosting_model.pkl')
neural_network_model = joblib.load('Neural_Network_model.pkl')
random_forest_model = joblib.load('Random_Forest_model.pkl')

# Precomputed averages from the training set
avg_self_reference_shares = 6401.697579821467
default_self_reference_min_shares = 3998.7553955201292
default_kw_avg_avg = 3135.8586389465236
default_kw_max_avg = 5657.211151064957
default_data_channel_is_world = 0.21256684491978609
default_LDA_02 = 0.21632096677306634
default_LDA_03 = 0.22376961651356772

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
    
    #processed_data = preprocess_data(input_data)
    # Process and predict
    processed_data = preprocess_data(input_data)
    print("Processed Manual Input Data:", processed_data)  # For debugging
    
    
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
        st.write("This dataset includes 39,644 rows and 61 features (including the target variable 'shares').")
        st.write("It is used for predicting the popularity of news articles based on various features like the number of words, images, and other metadata.")
        st.write("Most features are continuous variables (e.g., 'n_tokens_content', 'num_hrefs', 'data_channel_is_world').")
        st.write("The target variable, 'shares', is continuous but has been categorized for this analysis.")

    # Distribution Figures
    if st.button("Distributions"):
        st.write("Distributions of Selected Features")
        dist_image = Image.open('selected_features_distributions.png')
        st.image(dist_image, caption='Distributions of Selected Features', use_column_width=True)
        st.write("""
        This series of histograms displays the distribution of continuous features within the dataset. 
        Many features, such as the number of shares, average keyword weights (kw_avg_avg), and number of hyperlinks (num_hrefs), 
        show right-skewed distributions. This skewness indicates that while most articles have lower values for these features, 
        there are a few outliers with significantly higher values, likely due to a small number of highly popular or optimized articles.
        """)
        
    # Correlation Button: Display correlation heatmap
    if st.button("Correlation"):
        st.write("Correlation Matrix of Selected Features")
        image = Image.open('correlation_matrix.png')
        st.image(image, caption='Correlation Matrix', use_column_width=True)
        st.write("This heatmap shows the correlation between selected features. High values (closer to 1 or -1) indicate a strong relationship between two features.")
        st.write("For example, 'kw_avg_avg' and 'kw_max_avg' have a high positive correlation, indicating that as the average keyword weight increases, the maximum keyword average also tends to increase.")
        st.write("Negative correlations (values closer to -1) suggest an inverse relationship between features.")


    # Threshold Button: Display thresholds for categorization
    if st.button("Threshold"):
        with open('categorize_thresholds.txt', 'r') as file:
            thresholds = file.read()
        st.write("Thresholds for Categorization:")
        st.text(thresholds)

    # Outlier Visualization Before and After
    if st.button("Outliers"):
        st.write("Outlier Visualization - Before Capping")
        before_outlier_image = Image.open('beforeoutler.png')
        st.image(before_outlier_image, caption='Boxplot of Key Features Before Outlier Capping', use_column_width=True)
        
        st.write("Outlier Visualization - After Capping")
        after_outlier_image = Image.open('afteroutler.png')
        st.image(after_outlier_image, caption='Boxplot of Key Features After Outlier Capping', use_column_width=True)

        st.write("The boxplot helps identify potential outliers in the selected features.")
        st.write("Outlier thresholds based on 1st and 99th percentiles for each feature:")
        st.write("""
        - n_tokens_content: (0.0, 2256.14)
        - num_hrefs: (0.0, 56.0)
        - num_imgs: (0.0, 37.0)
        - shares: (381.0, 31657.0)
        """)
        st.write("Values outside these ranges are considered outliers and were capped during preprocessing to ensure robustness of the models.")

    # Evolution Button: Comparison of models' performance
    if st.button("Model Evolution"):
        st.write("Model Performance Comparison")
        results_df = pd.read_csv('final_model_evaluation_results.csv')
        st.write(results_df)

# Model Selection for prediction
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Gradient Boosting", "Neural Network", "Random Forest"])

# Make predictions based on the selected model
if 'processed_data' in locals():
    if model_choice == "Gradient Boosting":
        predictions = gradient_boosting_model.predict(processed_data)
    elif model_choice == "Neural Network":
        predictions = neural_network_model.predict(processed_data)
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
    if input_method in ["Manual Input", "Upload CSV"]:
        st.warning("Please upload a valid file or input correct data.")
    

