import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('/mount/src/diabetes/trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    elif prediction[0] == 1:
        return 'The person is prediabetic'
    else:
        return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    HighBP = st.selectbox('High Bp', [0, 1])
    HighChol = st.selectbox('HighChol', [0, 1])
    CholCheck = st.selectbox('CholCheck', [0, 1])
    BMI = st.text_input('BMI')
    Smoker = st.selectbox('Smoker', [0, 1])
    Stroke = st.selectbox('Stroke', [0, 1])
    HeartDiseaseorAttack = st.selectbox('HeartDiseaseorAttack', [0, 1])
    PhysActivity = st.selectbox('PhysActivity', [0, 1])
    Fruits = st.selectbox('Fruits', [0, 1])
    Veggies = st.selectbox('Veggies', [0, 1])
    HvyAlcoholConsump = st.selectbox('HvyAlcoholConsump', [0, 1])
    AnyHealthcare = st.selectbox('AnyHealthcare', [0, 1])
    NoDocbcCost = st.selectbox('NoDocbcCost', [0, 1])
    DiffWalk = st.selectbox('DiffWalk', [0, 1])
    Sex = st.selectbox('Sex', [0, 1])
    GenHlth = st.selectbox('GenHlth', [0, 1,2,3,4,5])
    Income = st.selectbox('Income', [0, 1,2,3,4,5,6,7,8])
    Education = st.selectbox('Education', [0, 1,2,3,4,5,6])
    MentHlth = st.selectbox('MentHlth', list(range(31)))
    PhysHlth = st.selectbox('PhysHlth', list(range(31)))
    
    
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
                                          Fruits, Veggies ,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,DiffWalk,Sex,GenHlth,Income
                                          ,Education,MentHlth, PhysHlth])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
