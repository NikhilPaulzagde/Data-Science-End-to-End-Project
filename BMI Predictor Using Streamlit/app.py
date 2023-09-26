from src.pipeline.prediction_pipeline import CustomClass,PredictionPipeline
import streamlit as st 

# Create a Streamlit form
st.title("Body Mass Index Form")


feet = st.sidebar.number_input("Enter Height in Feet", min_value=0.0, step=0.01)

# Convert feet to centimeters
centimeters = feet * 30.48  # 1 foot = 30.48 cm

# Display the result
st.sidebar.write(f"Height in Centimeters: {centimeters:.2f} cm")

# Add form elements for Gender, Height, and Weight
gender = st.radio("Gender", ("Male", "Female"))
height = int(st.number_input("Height (in cm)", min_value=0))
weight = int(st.number_input("Weight (in kg)", min_value=0))

# Display the submitted values
if st.button("Submit"):
    if gender == "Female":
        gender = 0

    else:
        gender = 1

    data = CustomClass (
        Gender= int(gender),
        Height= int(height),
        Weight= int(weight)
    )

    pred_df = data.get_dataframe()
    print(pred_df.dtypes)

    pp = PredictionPipeline()
   

    result = pp.get_predict(pred_df)
    print("Prediction: ",result[0])

    if result[0] == 0:
       st.error("Extremely Weak")
     
    elif result[0] == 1:
        st.warning("Weak")

    elif result[0] == 2:
        st.success("Normal")

    elif result[0] == 3:
        st.warning("Overweight")

    elif result[0] == 4:
        st.error("Obesity")

    else:
        st.error(" Extreme Obesity")


     
    