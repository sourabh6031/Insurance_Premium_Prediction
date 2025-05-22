import streamlit as st
import pandas as pd
import pickle

with open('scaled_scaler.pkl', 'rb') as std_scaler:
    scaler = pickle.load(std_scaler)

# Preprocessing function
def preprocess_input(input_data):
    """
    user_input: Dictionary containing feature values
    Returns: Scaled input data as a DataFrame
    """
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data],columns = ['Age','Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 
                'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries','any_disease', 'BMI'])

        # Apply scaling
        scaled_data = scaler.transform(input_df)

        # FIXING THE NAMES
        scaled_input = pd.DataFrame(scaled_data, columns=['Age','Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 
                'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries','any_disease', 'BMI'])

        return scaled_input

    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None