# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 01:49:19 2022

@author: user
"""

import numpy as np  #used for working with numpy arrays
import pickle       #used for loading the saved model
import streamlit as st  #used for deployment


#loading the save model
loaded_model=pickle.load(open('C:/Users/user/Desktop/CASE_STUDY_ML/trained_model.sav','rb'))

#create a function for prediction

def crop_prediction(input_data):
    
    #input_data = (104,18,30,23.603016,60.396475,6.779833,140.937041)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    return prediction

def main():
    
    #giving a title for 
    st.title('CROP RECOMMENDATION WEB APP')
    
    #getting input data from the user
    N = st.text_input('Enter the Nitrogen')
    P = st.text_input('Enter the phosporous') 
    K = st.text_input('Enter the potassium')
    temperature = st.text_input('Enter the temperature')
    humidity = st.text_input('Enter the humidity')
    ph = st.text_input('Enter the ph level')
    rainfall = st.text_input('Enter the rainfall')
    
    
    #code for predcition 
    crop = ''
    
    
    #creating a button for predicition
    if st.button('Crop Result'):
        crop=crop_prediction([N,P,K,temperature,humidity,ph,rainfall])
       # print("kunal")
        #st.write("Welcome to your first streamlit application"+crop)
        #print(crop)
        #st.text('You can grow '+crop)
        #st.text('You can grow '+np.array_str(crop))
    
   # crop1 = st.text_input('Result')
   # crop1="result:"+crop
    st.success('You can grow'+crop[0])
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    