import streamlit as st
import numpy as np
import xgboost as xgb

# Function to handle NA values and convert inputs to float
def parse_input(value):
    try:
        return float(value) if value not in ['', 'NA', 'na', None] else None
    except ValueError:
        return None
    
# Load your model
model = xgb.Booster()
model.load_model("fitted_model_final.model")

# Streamlit app title
st.title("XGBoost Model Prediction for Cardiac Biomarkers")

# Creating user input fields for the 3 specified biomarkers with allowance for NA values
troponin_i_input = st.text_input("Troponin I Value (enter 'NA' if not available)")
bnp_input = st.text_input("BNP Value (enter 'NA' if not available)")

# Convert inputs to float or None
troponin_i = parse_input(troponin_i_input)
bnp = parse_input(bnp_input)

# When the 'Predict' button is pressed
if st.button('Predict'):
    # Check if all values are NA
    if troponin_i is None and bnp is None:
        st.write("Please enter at least one biomarker value.")
    else:
        # Prepare the input data, handling NAs appropriately
        input_data = np.array([[troponin_i if troponin_i is not None else np.nan,
                                bnp if bnp is not None else np.nan]])

        # Mask for NA values to handle them in XGBoost
        mask = np.isnan(input_data)
        dmatrix = xgb.DMatrix(np.ma.masked_array(input_data, mask=mask))

        # Make prediction
        prediction = model.predict(dmatrix)

    # Determine the message and confidence based on the prediction value
    if prediction[0] > 0.475:
        confidence_percentage = round(prediction[0] * 100, 2)
        message = "GAD pattern is **unlikely** to be found on MRI screening."
    else:
        confidence_percentage = round((1 - prediction[0]) * 100, 2)
        message = "GAD pattern is **likely** to be found on MRI screening."

    # Display the prediction and the message
    #st.write(f"Prediction: {prediction[0]}")
    st.write(message)
    
    # Display 'confidence' using st.metric
    st.metric(label="Model's Confidence in Prediction", value=f"{confidence_percentage}%")
    
    
st.markdown("### Model Statistics")
st.markdown("""
- **Accuracy**: 76.67%
- **Precision**: 71.74%
- **Recall**: 80.49%
- **Specificity**: 73.47%
- **F1 Score**: 75.86%
- **NPV**: 81.82%
- **FPR**: 26.53%
- **AUC**: 87.70%
""")