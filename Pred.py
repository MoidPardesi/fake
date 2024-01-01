import numpy as np
import joblib
import streamlit as st

# loading the saved model
loaded_model = joblib.load(open('logistic_regression_model.sav', 'rb'))
loaded_vectorizer = joblib.load('count_vectorizer.sav')

# creating a function for Prediction
def news_prediction(headline, written_by, news):
    # Combine the features into a single string
    combined_features = f"{headline} {written_by} {news}"
    
    # Transform the input data using the loaded vectorizer
    transformed_input = loaded_vectorizer.transform([combined_features])
    
    # Make a prediction
    prediction = loaded_model.predict(transformed_input)
    
    if prediction[0] == 0:
        return 'This news is fake'
    else:
        return 'This news is real'

def main():
    # giving a title
    st.title('Fake News Prediction Web App')

    # getting the input data from the user
   
    headline = st.text_input('headline')
    written_by = st.text_input('written_by')
    news = st.text_input('news')

    # code for Prediction
    predict = ''

    # creating a button for Prediction
    if st.button('News Prediction Result'):
        predict = news_prediction([headline, written_by, news])

    st.success(predict)


if __name__ == '__main__':
    main()
