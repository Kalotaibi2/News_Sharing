import streamlit as st

st.title('News Popularity Predictor')

# User input for model selection
model_choice = st.selectbox('Choose a model', ['Naive Bayes', 'Logistic Regression'])

# Add a button for prediction
if st.button('Predict'):
    st.write(f'Using {model_choice} model for prediction...')
