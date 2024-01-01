import numpy as np
import joblib
import streamlit as st

# loading the saved model
loaded_model = joblib.load('logistic_regression_model.sav')
loaded_vectorizer = joblib.load('count_vectorizer.sav')

# creating a function for Prediction
def news_prediction(headline, written_by, news):
    # Combine the features into a single string
    combined_features = f"{headline} {written_by} {news}"
    
    # Transform the input data using the loaded vectorizer
    transformed_input = loaded_vectorizer.transform([combined_features])
    
    # Make a prediction
    prediction = loaded_model.predict(transformed_input)
    
    if prediction[1] == 1:
        return 'This news is real'
    else:
        return 'This news is fake'

def main():
    # giving a title
    st.title('Fake News Prediction Web App')
    
    # getting the input data from the user
    headline = st.text_input('Headline')
    written_by = st.text_input('Written by')
    news = st.text_area('News article text')  # Changed to text_area for longer input

    # code for Prediction
    prediction_message = ''

    # creating a button for Prediction
    if st.button('News Prediction Result'):
        prediction_message = news_prediction(headline, written_by, news)

    st.success(prediction_message)

if __name__ == '__main__':
    main()
