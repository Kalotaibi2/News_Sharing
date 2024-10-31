import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load dataset and models
data = pd.read_csv("OnlineNewsPopularity.csv")
data.columns = data.columns.str.strip().str.replace(" ", "_")  # Clean column names

# Define the expected columns globally
expected_columns = [
    'kw_avg_avg', 'LDA_03', 'kw_max_avg', 'self_reference_avg_sharess',
    'self_reference_min_shares', 'data_channel_is_world', 'LDA_02',
    'num_hrefs', 'num_imgs'
]

# Check if expected columns are in data
data = data[[col for col in expected_columns if col in data.columns]]  # Only keep selected features

# Load models
try:
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly_model.pkl')
    gradient_boosting_model = joblib.load('Gradient_Boosting_model.pkl')
    neural_network_model = joblib.load('Neural_Network_model.pkl')
    random_forest_model = joblib.load('Random_Forest_model.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to preprocess data
def preprocess_data(input_data_df):
    # Ensure columns match the expected set for scaler and poly
    try:
        data_scaled = scaler.transform(input_data_df)
        data_poly = poly.transform(data_scaled)
        return data_poly
    except ValueError as e:
        st.error("Data preprocessing error. Please check input data format.")
        st.write(e)
        return None

# Streamlit App
st.title("News Sharing Prediction App")
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input by ID", "Upload CSV", "View Preprocessing Results"])

# Model Selection for prediction
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Gradient Boosting", "Neural Network", "Random Forest"])

processed_data_manual = None

# Option 1: Manual Input with ID
if input_method == "Manual Input by ID":
    st.write("Enter an ID from 0 to 39643 to auto-fill features:")
    
    # ID Input Field
    id_input = st.number_input("Enter ID", min_value=0, max_value=39643, step=1)
    
    # Fetch selected record based on ID
    if 0 <= id_input < len(data):
        selected_record = data.iloc[int(id_input)].to_dict()
    
        # Display input fields for each selected feature
        st.write("Adjust or confirm feature values below:")
        input_data = {feature: st.number_input(f"Enter value for {feature}", value=selected_record[feature]) for feature in expected_columns}

        # Convert input_data to DataFrame for preprocessing
        processed_data_manual = preprocess_data(pd.DataFrame([input_data]))


# Option 2: Upload CSV
processed_data_csv = None
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Step 1: Read the uploaded file
        input_data = pd.read_csv(uploaded_file)
        
        # Step 2: Display the uploaded data preview
        st.write("Uploaded data preview:")
        st.write(input_data.head())
        input_data.columns = input_data.columns.str.strip()  # Ensure clean column names
        
        input_data = input_data[expected_columns]

        processed_data_csv = preprocess_data(input_data)

            
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

    # Outlier Visualization Before and After
    if st.button("Outliers"):
        st.write("Outlier Visualization - Before Capping")
        before_outlier_image = Image.open('beforeoutler.png')
        st.image(before_outlier_image, caption='Boxplot of Key Features Before Outlier Capping', use_column_width=True)
        
        st.write("Outlier Visualization - After Capping")
        after_outlier_image = Image.open('afteroutler.png')
        st.image(after_outlier_image, caption='Boxplot of Key Features After Outlier Capping', use_column_width=True)

    # Evolution Button: Comparison of models' performance
    if st.button("Model Evolution"):
        st.write("Model Performance Comparison")
        results_df = pd.read_csv('final_model_evaluation_results.csv')
        st.write(results_df)

# Model Selection for prediction
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model:", ["Gradient Boosting", "Neural Network", "Random Forest"])

# Make predictions based on the selected model
if input_method == "Manual Input by ID" and processed_data_manual is not None:
    
    if st.button("Predict") and processed_data is not None:
        if model_choice == "Gradient Boosting":
            predictions = gradient_boosting_model.predict(processed_data)
        elif model_choice == "Neural Network":
            predictions = neural_network_model.predict(processed_data)
        elif model_choice == "Random Forest":
            predictions = random_forest_model.predict(processed_data)
else:        
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
