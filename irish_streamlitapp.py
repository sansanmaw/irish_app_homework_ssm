import streamlit as st
import pickle
import numpy as np

# Streamlit UI

st.title("Irish Prediction App (San San Maw)")
st.write("Enter infomation details to predict species.")

# Input fields


sepal_length = st.number_input ("Sepal Length (in cm)", min_value = 4.0, max_value=8.0)
sepal_width = st.number_input ("Sepal Width (in cm)", min_value = 1.0, max_value=5.0)
petal_length = st.number_input ("Petal Length (in cm)", min_value = 1.0, max_value=5.0)
petal_width = st.number_input ("Petal Width (in cm)", min_value = 0.1, max_value=3.0)

with open('knn_irish.pkl','rb') as f:

    loaded_model=pickle.load(f)

with open('knn_scaler_irish.pkl','rb') as f:

    loaded_scaler=pickle.load(f)

if st.button("Predict"):

    input_features=np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    input_features = loaded_scaler.transform(input_features) 
  
    value = loaded_model.predict(input_features)[0]  

    st.write(f"Predicted value (raw): {value}") 
    st.success(f" Predicted Species: {value}")
