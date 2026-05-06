import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIG & STYLING
st.set_page_config(
    page_title="SmartHire",  
    page_icon="🎯",               
    layout="centered"
)


st.title("🎯 Smart Hire: Performance Prediction Engine")    
st.caption("Pre-Hire Performance Prediction Engine") # Added a cool subtitle!
st.markdown("""
Identify statistically likely **Outstanding performers** before they are hired using 6 key pre-hire indicators.
""")

# 2. LOAD THE PRODUCTION MODEL (Safe Mode)

try:
    model_path = 'C:\Users\Administrator\Desktop\models\final_hiring_model.pkl' 
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model. Check the file path: {e}")
    st.stop() 


# 3. CREATE THE USER INPUT FORM (Sidebar)
st.sidebar.header("📝 Applicant Information")

# Input 1: Age
age = st.sidebar.slider("1. Applicant Age", 18, 60, 30)

# Input 2: Total Work Experience
experience = st.sidebar.slider("2. Total Work Experience (Years)", 0, 40, 5)

# Input 3: Number of Past Companies
companies = st.sidebar.slider("3. Number of Past Companies Worked For", 0, 9, 2)

# Input 4: Education Level (Mapped to 1-5)
edu_options = ["Below College", "College", "Bachelor's", "Master's", "Doctorate"]
edu_level_map = {"Below College": 1, "College": 2, "Bachelor's": 3, "Master's": 4, "Doctorate": 5}
edu_choice = st.sidebar.selectbox("4. Highest Education Level", edu_options)
edu_level = edu_level_map[edu_choice]

# Input 5: Business Travel Frequency (Mapped to 0-2)
travel_options = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
travel_choice = st.sidebar.selectbox("5. Willingness to Travel", travel_options)
travel_freq = travel_map[travel_choice]

# Input 6: Education Background
edu_background_options = ["Life Sciences", "Medical", "Marketing", "Other", "Technical Degree"]
edu_background = st.sidebar.selectbox("6. Field of Study", edu_background_options)


# 4. PROCESS INPUTS & PREDICT
if st.sidebar.button("🚀 Predict Performance", type="primary"):
    
    with st.spinner('Analyzing applicant profile...'):
        
        # A. Create a dictionary of the raw inputs
        input_dict = {
            'Age': age,
            'TotalWorkExperienceInYears': experience,
            'NumCompaniesWorked': companies,
            'EmpEducationLevel': edu_level,
            'BusinessTravelFrequency': travel_freq,
            'EducationBackground': edu_background
        }
        
        # B. Convert to DataFrame
        applicant_df = pd.DataFrame([input_dict])
        
        # C. One-Hot Encode the Education Background (exactly like training)
        applicant_encoded = pd.get_dummies(applicant_df, columns=['EducationBackground'], dtype=int)
        
        # D. Align columns with the model's exact expectations
        required_features = model.feature_names_in_
        applicant_final = applicant_encoded.reindex(columns=required_features, fill_value=0)
        
        # E. Make the prediction
        prediction = model.predict(applicant_final)
        probability = model.predict_proba(applicant_final)[0][1] * 100
  
   
    # 5. DISPLAY THE RESULTS
    st.subheader("📊 Prediction Result")
    st.markdown("---")
    
    if prediction[0] == 1:
        st.success(f"🎉 **Prediction: OUTSTANDING PERFORMER**")
        st.info(f"The model is **{probability:.1f}% confident** this candidate will be a top-tier performer. Highly recommend advancing to the human interview stage.")
    else:
        st.warning(f"📉 **Prediction: AVERAGE PERFORMER**")
        st.info(f"The model is **{100 - probability:.1f}% confident** this candidate falls into the standard performance bracket. Proceed with standard screening.")
        
    # Show the mathematical breakdown
    with st.expander("🔍 View Technical Details"):
        st.write("**Applicant Data passed to model:**")
        st.dataframe(applicant_final.T)
        st.write(f"**Raw Probability Score for Class 1:** {probability/100:.4f}")

else:
    st.info("👈 Adjust the applicant's details in the sidebar and click **Predict Performance** to see the result.")