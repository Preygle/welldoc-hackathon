'''
Streamlit Frontend for Health Prediction
'''
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import re
import pickle
import xgboost as xgb

# Load the trained models
with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("hypertension_model.pkl", "rb") as f:
    hypertension_model = pickle.load(f)
with open("heart_disease_model.pkl", "rb") as f:
    heart_disease_model = pickle.load(f)

st.set_page_config(layout="wide")
st.title('Health Risk Prediction AI')

st.info("This application predicts the risk of Diabetes, Heart Disease, and Hypertension based on user inputs. The predictions are made by XGBoost models.")

# IMPORTANT: Replace with your actual Gemini API Key
# To get an API key, visit https://makersuite.google.com/
gemini_api_key = "AIzaSyAEDFRrwIO_P6aOHHw7opzmvBx9s8UgwDg"

with st.form("prediction_form"):
    st.header('Patient Information')

    # User Inputs laid out in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=0, max_value=120, value=45)
        height_cm = st.number_input('Height (cm)', min_value=50, max_value=250, value=170, help="Needed to calculate BMI.")
        weight_kg = st.number_input('Weight (kg)', min_value=10, max_value=300, value=75, help="Needed to calculate BMI.")
        smoking_status_str = st.selectbox('Smoking Habit', ['Never', 'Former', 'Current', 'Ever', 'Not Current'], index=0)

    with col2:
        glucose_level = st.number_input('Glucose Level (mg/dL)', min_value=50, max_value=500, value=100)
        sugar_level = st.number_input('Sugar Level (Fasting, mg/dL)', min_value=50, max_value=500, value=90)
        cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
        heart_rate = st.number_input('Heart Rate (bpm)', min_value=40, max_value=200, value=72)
        drinks_per_week = st.number_input('Alcoholic Drinks per Week', min_value=0, max_value=50, value=0)

    with col3:
        systolic_bp = st.number_input('Systolic BP (High BP value)', min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input('Diastolic BP (Low BP value)', min_value=40, max_value=150, value=80)
        sleep_hours = st.number_input('Average Sleep Hours per Night', min_value=0.0, max_value=24.0, value=7.5, step=0.5)
        step_count = st.number_input('Average Daily Step Count', min_value=0, max_value=30000, value=8000)
        stress_rating = st.slider('Stress Rating (1-10)', min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("Predict Health Risks & Get AI Suggestions")

if submitted:
    # --- 1. Collect and Encode Inputs ---
    gender_encoded = 1 if gender == 'Male' else 0
    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2, 'Ever': 3, 'Not Current': 4}
    smoking_encoded = smoking_map[smoking_status_str]

    # --- 2. Engineer Features ---
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2) if height_m > 0 else 0

    if step_count < 5000:
        activity_level_encoded = 0  # Sedentary
    elif 5000 <= step_count < 10000:
        activity_level_encoded = 1  # Moderately Active
    else:
        activity_level_encoded = 2  # Active

    alcohol_use_encoded = 1 if drinks_per_week > 0 else 0

    binge_flag_encoded = 0
    if (gender == 'Female' and drinks_per_week >= 4) or (gender == 'Male' and drinks_per_week >= 5):
        binge_flag_encoded = 1

    high_stress_encoded = 1 if stress_rating > 7 else 0
    poor_sleep_encoded = 1 if sleep_hours < 7 else 0
    stress_x_poor_sleep_encoded = high_stress_encoded * poor_sleep_encoded

    # --- 3. Assemble Feature Vector ---
    feature_cols_ordered = [
        'gender', 'age', 'bmi', 'glucose level', 'heart rate', 'sleep hours', 'sugar level', 'cholesterol',
        'systolic_bp', 'diastolic_bp', 'smoking_status', 'Step_count', 'drinks_per_week',
        'stress_rating_1_to_10', 'Activity_Level', 'Alcohol_Use', 'Binge_Flag', 'High_Stress',
        'Poor_Sleep', 'Stress_x_Poor_Sleep'
    ]

    user_data = {
        'gender': gender_encoded, 'age': age, 'bmi': bmi, 'glucose level': glucose_level, 'heart rate': heart_rate,
        'sleep hours': sleep_hours, 'sugar level': sugar_level, 'cholesterol': cholesterol, 'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp, 'smoking_status': smoking_encoded, 'Step_count': step_count,
        'drinks_per_week': drinks_per_week, 'stress_rating_1_to_10': stress_rating,
        'Activity_Level': activity_level_encoded, 'Alcohol_Use': alcohol_use_encoded, 'Binge_Flag': binge_flag_encoded,
        'High_Stress': high_stress_encoded, 'Poor_Sleep': poor_sleep_encoded,
        'Stress_x_Poor_Sleep': stress_x_poor_sleep_encoded
    }

    input_df = pd.DataFrame([user_data])[feature_cols_ordered]

    # --- 4. Make Predictions ---
    diabetes_pred = diabetes_model.predict(input_df)[0]
    diabetes_prob = diabetes_model.predict_proba(input_df)[0][1]
    hypertension_pred = hypertension_model.predict(input_df)[0]
    hypertension_prob = hypertension_model.predict_proba(input_df)[0][1]
    heart_disease_pred = heart_disease_model.predict(input_df)[0]
    heart_disease_prob = heart_disease_model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Results")
    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        st.metric("Diabetes Risk", f"{diabetes_prob:.0%}")
        if diabetes_pred == 1:
            st.error("High Risk for Diabetes")
        else:
            st.success("Low Risk for Diabetes")

    with pred_col2:
        st.metric("Heart Disease Risk", f"{heart_disease_prob:.0%}")
        if heart_disease_pred == 1:
            st.error("High Risk for Heart Disease")
        else:
            st.success("Low Risk for Heart Disease")

    with pred_col3:
        st.metric("Hypertension Risk", f"{hypertension_prob:.0%}")
        if hypertension_pred == 1:
            st.error("High Risk for Hypertension")
        else:
            st.success("Low Risk for Hypertension")

    st.markdown("---")

    # --- 5. Generate AI-Powered Suggestions ---
    st.subheader("AI-Powered Suggestions")
    if gemini_api_key == "YOUR_API_KEY":
        st.warning("Please replace 'YOUR_API_KEY' in the code with your actual Gemini API Key to generate suggestions.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            activity_level_str = "Sedentary" if activity_level_encoded == 0 else "Moderately Active" if activity_level_encoded == 1 else "Active"
            
            prompt = f'''
You are an expert health and wellness coach AI. Your task is to provide personalized, actionable, and supportive suggestions based on a user's health data.

**User's Health Profile:**
- **Age:** {age}
- **Gender:** {gender}
- **BMI:** {bmi}
- **Glucose Level:** {glucose_level} mg/dL
- **Fasting Sugar Level:** {sugar_level} mg/dL
- **Cholesterol:** {cholesterol} mg/dL
- **Heart Rate:** {heart_rate} bpm
- **Blood Pressure:** {systolic_bp}/{diastolic_bp} mmHg
- **Smoking Habit:** {smoking_status_str}
- **Alcohol Consumption:** {drinks_per_week} drinks per week
- **Average Sleep:** {sleep_hours} hours/night
- **Daily Steps:** {step_count}
- **Stress Level (1-10):** {stress_rating}

**Derived Insights:**
- **Activity Level:** {activity_level_str}
- **Binge Drinking:** {"Yes" if binge_flag_encoded == 1 else "No"}
- **High Stress:** {"Yes" if high_stress_encoded == 1 else "No"}
- **Poor Sleep:** {"Yes" if poor_sleep_encoded == 1 else "No"}

**Your Instructions:**

1.  **Start with a brief, encouraging summary** of the user's overall health profile based on the provided data.
2.  **Provide specific, numbered suggestions** in the following categories: Diet, Exercise, and Lifestyle.
3.  **Be empathetic and non-judgmental.** Use a positive and encouraging tone.
4.  **Prioritize the most critical areas.** If blood pressure is very high, that should be a primary focus.
5.  **Acknowledge healthy habits.** If the user has a good sleep schedule or is a non-smoker, praise them for it.
6.  **Keep advice practical and easy to implement.** For example, instead of "exercise more," suggest "try a 15-minute brisk walk after dinner."
7.  **Identify and flag critical conditions.** If you identify a critical health risk, wrap your suggestion for it in `[CRITICAL]` and `[/CRITICAL]` tags.
8.  **Include a clear disclaimer** at the end that this is not medical advice and the user should consult a healthcare professional.

**Critical Conditions to watch for:**
- BMI < 18.5 or BMI >= 25
- Sleep hours <= 4
- Systolic BP >= 160 or Diastolic BP >= 100
- Glucose Level >= 180
- Heart Rate > 100 or < 50 (at rest)

**Example of a critical suggestion:**
`[CRITICAL]Your BMI is quite high, which puts you at a significant risk for several health issues. It is strongly recommended to consult with a doctor to create a safe and effective plan. In the meantime, consider starting with a short, 10-minute walk each day.[/CRITICAL]`

**Generate the response now.**
'''

            with st.spinner("Generating personalized suggestions with Gemini..."):
                response = model.generate_content(prompt)
                text = response.text

                critical_suggestions = re.findall(r"\[CRITICAL\](.*?)\[/CRITICAL\]", text, re.DOTALL)
                
                if critical_suggestions:
                    st.subheader("Critical Alerts")
                    for suggestion in critical_suggestions:
                        st.error(suggestion.strip())

                non_critical_text = re.sub(r"\[CRITICAL\].*?\[/CRITICAL\]", "", text, flags=re.DOTALL)
                
                st.subheader("Personalized Suggestions")
                st.markdown(non_critical_text.strip())

        except Exception as e:
            st.error(f"An error occurred while generating suggestions: {e}")


    with st.expander("View Generated Model Input"):
        st.write("The following data was generated from your inputs and will be fed into the models.")
        st.dataframe(input_df)