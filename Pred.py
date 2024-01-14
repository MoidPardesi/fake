import numpy as np
import joblib
import streamlit as st


loaded_model = joblib.load('logistic_regression_model.sav')
loaded_vectorizer = joblib.load('count_vectorizer.sav')


def news_prediction(headline, written_by, news):
    combined_features = f"{headline} {written_by} {news}"
    transformed_input = loaded_vectorizer.transform([combined_features])
    prediction = loaded_model.predict(transformed_input)
    
    if prediction[0] == 0:
        return 'This news is fake'
    else:
        return 'This news is real'

def main():
    
    st.title('Fake News Prediction Web App')
    headline = st.text_input('Headline')
    written_by = st.text_input('Written by')
    news = st.text_area('News article text')  # Changed to text_area for longer input

   
    prediction_message = ''

    if st.button('News Prediction Result'):
        if headline and written_by and news:
            prediction_message = news_prediction(headline, written_by, news)
        else :
            st.error("Please fill in all the fields to get the prediction.")

    st.success(prediction_message)

if __name__ == '__main__':
    main()
