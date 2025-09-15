import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import re
import pickle
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

# Load the trained models
with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("hypertension_model.pkl", "rb") as f:
    hypertension_model = pickle.load(f)
with open("heart_disease_model.pkl", "rb") as f:
    heart_disease_model = pickle.load(f)

st.set_page_config(layout="wide")
st.title("GlucoGuard's Health Risk Prediction AI")

st.info("This application predicts the risk of Diabetes, Heart Disease, and Hypertension based on user inputs. The predictions are made by XGBoost models.")
gemini_api_key = os.getenv("GEMINI_API_KEY")

with st.form("prediction_form"):
    st.header('Patient Information')

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=0, max_value=120, value=45)
        height_cm = st.number_input(
            'Height (cm)', min_value=50, max_value=250, value=170, help="Needed to calculate BMI.")
        weight_kg = st.number_input(
            'Weight (kg)', min_value=10, max_value=300, value=75, help="Needed to calculate BMI.")
        smoking_status_str = st.selectbox(
            'Smoking Habit', ['Never', 'Former', 'Current', 'Ever', 'Not Current'], index=0)

    with col2:
        glucose_level = st.number_input(
            'Glucose Level (mg/dL)', min_value=50, max_value=500, value=100)
        sugar_level = st.number_input(
            'Sugar Level (Fasting, mg/dL)', min_value=50, max_value=500, value=90)
        cholesterol = st.number_input(
            'Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
        heart_rate = st.number_input(
            'Heart Rate (bpm)', min_value=40, max_value=200, value=72)
        drinks_per_week = st.number_input(
            'Alcoholic Drinks per Week', min_value=0, max_value=50, value=0)

    with col3:
        systolic_bp = st.number_input(
            'Systolic BP (High BP value)', min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input(
            'Diastolic BP (Low BP value)', min_value=40, max_value=150, value=80)
        sleep_hours = st.number_input(
            'Average Sleep Hours per Night', min_value=0.0, max_value=24.0, value=7.5, step=0.5)
        step_count = st.number_input(
            'Average Daily Step Count', min_value=0, max_value=30000, value=8000)
        stress_rating = st.slider(
            'Stress Rating (1-10)', min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button(
        "Predict Health Risks & Get AI Suggestions")

if submitted:
    # Collect Inputs
    gender_encoded = 1 if gender == 'Male' else 0
    smoking_map = {'Never': 0, 'Former': 1,
                   'Current': 2, 'Ever': 3, 'Not Current': 4}
    smoking_encoded = smoking_map[smoking_status_str]

    # Engineer Features
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

    # Assemble Feature Vector
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

    # Make Predictions
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

    # Generate AI-Powered Suggestions
    st.subheader("AI-Powered Personalized Suggestions")
    if not gemini_api_key:
        st.warning(
            "Please add your Gemini API Key to the .env file to generate suggestions.")
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

            1.  **Provide 2-3 concise, actionable bullet-point suggestions** for each of the following categories: `==DIET==`, `==EXERCISE==`, and `==LIFESTYLE==`.
            2.  **Start each category with the corresponding header** (e.g., `==DIET==`).
            3.  **Be empathetic and non-judgmental.** Use a positive and encouraging tone.
            4.  **Prioritize the most critical areas.**
            5.  **Acknowledge healthy habits.**
            6.  **Keep advice practical and easy to implement.**
            7.  **Identify and flag critical conditions.** If you identify a critical health risk, wrap your suggestion for it in `[CRITICAL]` and `[/CRITICAL]` tags.
            8.  **No need to generate "Remember, small, consistent changes can lead to significant improvements in your overall health and wellness. Keep up the fantastic work!"
            '''

            with st.spinner("Generating personalized suggestions with Gemini..."):
                response = model.generate_content(prompt)
                text = response.text

                # --- Critical Alerts ---
                critical_suggestions = re.findall(
                    r"\[CRITICAL\](.*?)\[/CRITICAL\]", text, re.DOTALL | re.IGNORECASE)
                if critical_suggestions:
                    st.subheader("🚨 Critical Alerts")
                    for suggestion in critical_suggestions:
                        st.error(suggestion.strip())

                # Remove critical tags for main display
                text = re.sub(
                    r"\[CRITICAL\](.*?)\[/CRITICAL\]",
                    r"\1",
                    text,
                    flags=re.DOTALL | re.IGNORECASE
                )

                diet_suggestions = re.search(
                    r"==DIET==\n(.*?)\n==EXERCISE==", text, re.DOTALL)
                exercise_suggestions = re.search(
                    r"==EXERCISE==\n(.*?)\n==LIFESTYLE==", text, re.DOTALL)
                lifestyle_suggestions = re.search(
                    r"==LIFESTYLE==\n(.*)", text, re.DOTALL)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 🍎 Diet")
                    if diet_suggestions:
                        st.markdown(diet_suggestions.group(1).strip())
                    else:
                        st.markdown(
                            "No specific diet suggestions at this time.")

                with col2:
                    st.markdown("### 🏃‍♂️ Exercise")
                    if exercise_suggestions:
                        st.markdown(exercise_suggestions.group(1).strip())
                    else:
                        st.markdown(
                            "No specific exercise suggestions at this time.")

                with col3:
                    st.markdown("### 🧘‍♀️ Lifestyle")
                    if lifestyle_suggestions:
                        st.markdown(lifestyle_suggestions.group(1).strip())
                    else:
                        st.markdown(
                            "No specific lifestyle suggestions at this time.")

        except Exception as e:
            st.error(f"An error occurred while generating suggestions: {e}")

    with st.expander("View Generated Model Input"):
        st.write(
            "The following data was generated from your inputs and will be fed into the models.")
        st.dataframe(input_df)
