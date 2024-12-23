import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load dataset and models
data = pd.read_csv("OnlineNewsPopularity.csv")
data.columns = data.columns.str.strip()  # Clean column names

# Define the expected columns globally
expected_columns = [
    'kw_avg_avg', 'LDA_03', 'kw_max_avg', 'self_reference_avg_sharess',
    'self_reference_min_shares', 'data_channel_is_world', 'LDA_02',
    'num_hrefs', 'num_imgs'
]

#Filter dataset to only include expected features
data = data[[col for col in expected_columns if col in data.columns]]  # Only keep selected features

# Load scaler, polynomial transformation, and prediction models
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_model.pkl')
gradient_boosting_model = joblib.load('Gradient_Boosting_model.pkl')
neural_network_model = joblib.load('Neural_Network_model.pkl')
random_forest_model = joblib.load('Random_Forest_model.pkl')

# Function to preprocess data: applies scaling and polynomial transformation
def preprocess_data(input_data_df):
    data_scaled = scaler.transform(input_data_df)
    data_poly = poly.transform(data_scaled)
    return data_poly

# Streamlit app title and sidebar input options
st.title("News Sharing Prediction App")
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input by ID", "Upload CSV", "View Preprocessing Results"])

# Variables to hold processed data for different input methods
processed_data_manual = None
processed_data_csv = None

# Option 1: Manual Input with ID
if input_method == "Manual Input by ID":
    st.write("Enter an ID from 0 to 39643 to auto-fill features:")
    
    # ID Input Field for selecting a record by ID
    id_input = st.number_input("Enter ID", min_value=0, max_value=39643, step=1)
    
    # Fetch selected record based on ID
    if 0 <= id_input < len(data):
        selected_record = data.iloc[int(id_input)].to_dict()
    
        # Display input fields for each selected feature with default values
        st.write("Adjust or confirm feature values below:")
        input_data = {feature: st.number_input(f"Enter value for {feature}", value=selected_record[feature]) for feature in expected_columns}

        # Convert input_data to DataFrame for preprocessing
        processed_data_manual = preprocess_data(pd.DataFrame([input_data]))


#Option 2: Upload CSV for bulk predictions
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Step 1: Read the uploaded file
        input_data = pd.read_csv(uploaded_file)
        
        # Step 2: Display the uploaded data preview
        st.write("Uploaded data preview:")
        st.write(input_data.head())
        input_data.columns = input_data.columns.str.strip()  # Ensure clean column names
        
        # Filter columns to include only expected features and preprocess
        input_data = input_data[expected_columns]
        processed_data_csv = preprocess_data(input_data)

            
# Option 3: View Preprocessing Results
elif input_method == "View Preprocessing Results":
    st.subheader("Preprocessing Steps and Visualizations")

    # Overview Button: Display basic dataset information
    if st.button("Overview"):
        data = pd.read_csv("OnlineNewsPopularity.csv")
        st.write("Dataset Overview:")
        st.write(data.head())
        st.write("This dataset includes 39,644 rows and 61 features (including the target variable 'shares').")
        st.write("It is used for predicting the popularity of news articles based on various features like the number of words, images, and other metadata.")
        st.write("All features are continuous variables (e.g., 'n_tokens_content', 'num_hrefs', 'data_channel_is_world').")
        st.write("The target variable, 'shares', is continuous but has been categorized for this analysis.")

    # Display distribution figures for selected features
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
        
    # Display correlation heatmap of selected features
    if st.button("Correlation"):
        st.write("Correlation Matrix of Selected Features")
        image = Image.open('correlation_matrix.png')
        st.image(image, caption='Correlation Matrix', use_column_width=True)
        st.write("This heatmap shows the correlation between selected features. High values (closer to 1 or -1) indicate a strong relationship between two features.")
        st.write("For example, 'kw_avg_avg' and 'kw_max_avg' have a high positive correlation, indicating that as the average keyword weight increases, the maximum keyword average also tends to increase.")
        st.write("Negative correlations (values closer to -1) suggest an inverse relationship between features.")

    # Display thresholds used for categorization in the dataset
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

    # Display model performance comparison
    if st.button("Model Evolution"):
        st.write("Model Performance Comparison")
        results_df = pd.read_csv('final_model_evaluation_results.csv')
        st.write(results_df)

# Model Selection for prediction
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Gradient Boosting", "Neural Network", "Random Forest"])

# Model prediction based on input method and processed data
if input_method == "Manual Input by ID" and processed_data_manual is not None:
    # Single record prediction with "Predict" button for manual input
    if st.button("Predict") and processed_data_manual is not None:
        if model_choice == "Gradient Boosting":
            predictions = gradient_boosting_model.predict(processed_data_manual)
        elif model_choice == "Neural Network":
            predictions = neural_network_model.predict(processed_data_manual)
        elif model_choice == "Random Forest":
            predictions = random_forest_model.predict(processed_data_manual)
        
        # Map numerical predictions to share categories
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        predicted_categories = [category_map[pred] for pred in predictions]
        st.write(f"Predicted Share Category: {predicted_categories[0]}")
    else:
        st.warning("Please input correct data.")

elif input_method == "Upload CSV" and processed_data_csv is not None:    
    # Automatically predict for all records in uploaded CSV  
    if 'processed_data_csv' in locals():
        if model_choice == "Gradient Boosting":
            predictions = gradient_boosting_model.predict(processed_data_csv)
        elif model_choice == "Neural Network":
            predictions = neural_network_model.predict(processed_data_csv)
        elif model_choice == "Random Forest":
            predictions = random_forest_model.predict(processed_data_csv)

        # Map predictions to share categories
        category_map = {0: "Low", 1: "Medium", 2: "High"}
        predicted_categories = [category_map[pred] for pred in predictions]

        # Display predictions for all records
        if len(predicted_categories) == 1:
            st.write(f"Predicted Share Category: {predicted_categories[0]}")
        else:
            st.write("Predicted Share Categories:")
            st.write(predicted_categories)
else:
    if input_method == "Upload CSV"and processed_data_csv is None:
        st.warning("Please upload a valid file")
