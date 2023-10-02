import streamlit as st
from src.pipeline.prediction_pipeline import Customclass,PredictionPipeline

# Create a Streamlit form
st.title("User Information Form")

# Input fields for user information
age = st.number_input("Age", min_value=0, max_value=150, step=1)
gender = st.radio("Gender", ["Male", "Female"])
income = st.number_input("Income", min_value=0, step=1000)
education = st.selectbox("Education", ["High School Diploma","Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
num_children = st.number_input("Number of Children", min_value=0, step=1)
home_ownership = st.radio("Home Ownership", ["Owned", "Rented"])

# Display the submitted values
if st.button("Submit"):
    
    data = Customclass(
        Age= int(age),
        Gender= gender,
        Income= int(income),
        Education= education,
        Marital_Status= marital_status,
        Number_of_Children= int(num_children),
        Home_Ownership= home_ownership
    )

    final_data = data.get_dataframe()
    predict_pipeline = PredictionPipeline()
    pred = predict_pipeline.predict(final_data)
        
    result = int(pred[0])
    print("Prediction: ",result)

    if result[0]==0:
        st.error("Credit Score is Low")

    elif result[0]==1:
        st.warning("Credit Score is Average")

    else:
        st.success("Credit Score is High")