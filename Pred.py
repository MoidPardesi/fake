import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('logistic_regression_model.sav', 'rb'))

# creating a function for Prediction
def news_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Ensure loaded_model is a model with a predict method
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

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
