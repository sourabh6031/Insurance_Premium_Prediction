import streamlit as st
import pandas as pd
import xgboost 
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

from preprocess import preprocess_input

# Streamlit App Title
st.header("Insurance Premium Prediction app💲")

st.divider() # for dividing line

st.text("Check your estimated insurance premium amount.")
df = pd.read_csv('insurance_file.csv')
st.write("### Sample preview of the values")
df = df.iloc[:,1:]
st.dataframe(df.head())

# IMPORTING TRAINED XGBOOST MODEL
try:
    with open('best_xgb_grid_model.pkl','rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error in loading XGB trained model: {e}")

# IMPORTING TRAINED STANDARD SCALER
# IMPORTING DONE IN 'preprocess'


st.divider()

st.subheader("Please fill in the data. :sunglasses:")


col1,col2, col3 = st.columns(3)
Age = col1.number_input("Input numerical value for Age")
Diabetes = col2.selectbox("Diabetes",[0,1])
BloodPressureProblems = col3.selectbox("Blood Pressure Problems",[0,1])
AnyTransplants = col1.selectbox("Any Transplants",[0,1])
AnyChronicDiseases = col2.selectbox("ChronicDiseases",[0,1])
KnownAllergies = col3.selectbox("KnownAllergies",[0,1])
HistoryOfCancerInFamily = col1.selectbox("HistoryOfCancerInFamily",[0,1])
NumberOfMajorSurgeries = col2.selectbox("Number of Surgeries",[0,1,2,3])
Height = col3.number_input("Input numerical value for Height")
Weight = col1.number_input("Input numerical value for Weight")



# FEATURE ENGINEERING
disease_columns = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 
                'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries' ]

any_disease = 1 if any([Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, KnownAllergies, HistoryOfCancerInFamily]) else 0    # creating 'any_disease'

BMI = Weight / ((Height / 100) ** 2) if Height != 0 else 0      # creating 'BMI'

input_data = [Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries, any_disease, BMI]

# PREDICTION
if st.button("Predict the Premium Amount."):
    processed_data = preprocess_input(input_data)
    
    st.subheader("Your Premium amount should be around.")
    
    premium_pred = model.predict(processed_data)
    amount = np.round(premium_pred,0)
    st.header(amount[0])
    st.balloons()
