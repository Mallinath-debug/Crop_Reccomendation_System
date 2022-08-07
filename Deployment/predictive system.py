# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the save model
loaded_model=pickle.load(open('C:/Users/user/Desktop/CASE_STUDY_ML/trained_model.sav','rb'))

input_data = (104,18,30,23.603016,60.396475,6.779833,140.937041)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)