import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Loading the saved model
try:
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: 'trained_model.sav' not found. Please ensure the trained model is saved in the same directory.")
    st.stop()


st.title('Sonar Data Prediction App')

st.write("Enter the 60 features of the sonar data to get a prediction (Rock 'R' or Mine 'M').")

# Create input fields for the 60 features
# You can adjust the range and step based on your data's characteristics
input_data = []
for i in range(60):
    input_data.append(st.number_input(f'Feature {i+1}', value=0.0, key=f'feature_{i}')) # Added a unique key

# Create a button to trigger prediction
if st.button('Predict'):
    # Convert the input data to a numpy array and reshape it
    input_data_as_numpy_array = np.asarray(input_data)
    # Ensure the input has the correct number of features (60)
    if input_data_as_numpy_array.shape[0] != 60:
        st.error(f"Error: Expected 60 features but received {input_data_as_numpy_array.shape[0]}. Please enter all feature values.")
    else:
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        # Display the prediction
        if (prediction[0] == 'R'):
            st.success('The object is a Rock')
        else:
            st.success('The object is a Mine')
