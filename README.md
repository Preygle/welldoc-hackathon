# GlucoGuard's Health Risk Prediction AI

This project is a web-based application that predicts a user's risk for three chronic conditions: Diabetes, Hypertension, and Heart Disease. It uses machine learning models trained on a custom-engineered dataset and provides personalized health and wellness suggestions using Google's Gemini AI.

The application is built with Python, using XGBoost for the prediction models and Streamlit for the interactive user interface.

## Live Demo

The deployed Streamlit application can be accessed here: [health-risk-prediction-ai.streamlit.app/](https://health-risk-prediction-ai.streamlit.app/)

## Features

- **Chronic Disease Prediction:** Predicts the probability of developing Diabetes, Hypertension, and Heart Disease.
- **Interactive UI:** A user-friendly web interface built with Streamlit to input health metrics.
- **Personalized AI Suggestions:** Leverages the Gemini 1.5 Flash model to provide actionable advice on diet, exercise, and lifestyle based on the user's data.
- **Dynamic Feature Engineering:** Calculates BMI and derives behavioral features in real-time from user inputs.
- **Multiple Models:** Utilizes three distinct XGBoost classification models, one for each medical condition.

## How It Works

The application follows a simple workflow:
1.  **Data Input:** The user enters their health information (e.g., age, BMI, blood pressure, glucose levels) into a web form.
2.  **Feature Engineering:** The application processes the raw inputs to create a feature vector that matches the format used for model training. This includes calculating BMI, encoding categorical values (like gender and smoking status), and deriving risk flags (like `High_Stress` or `Poor_Sleep`).
3.  **Prediction:** The engineered feature set is fed into the three pre-trained XGBoost models.
4.  **Display Results:** The models output a risk probability for each of the three conditions (Diabetes, Hypertension, Heart Disease), which is displayed to the user.
5.  **Generate Suggestions:** The user's data and risk profile are sent to the Gemini AI, which generates personalized and actionable health recommendations.

## File Structure & Scripts

-   `app.py`: The main script that runs the Streamlit web application.
-   `create_models.py`: This script trains the XGBoost models for each chronic condition using the final dataset (`final.csv`) and saves them as `.pkl` files.
-   `add_features.py`: Performs feature engineering by adding synthetic and derived columns to the initial dataset.
-   `encode_features.py`: Encodes categorical features (e.g., 'gender', 'Activity_Level') into numerical values suitable for the ML models.
-   `train_models.py`: An auxiliary script used for experimenting with and comparing different classification models (XGBoost, Random Forest, Logistic Regression).
-   `*.pkl`: The serialized, pre-trained XGBoost model files (`diabetes_model.pkl`, `hypertension_model.pkl`, `heart_disease_model.pkl`).
-   `*.csv`: The datasets used in the project. `final.csv` is the fully processed dataset used for training.

## Dataset & Feature Engineering

The training data (`final.csv`) was created by augmenting and combining several public datasets from Kaggle and then engineering new features.

### Original Datasets
- [WHOOP Dataset](https://www.kaggle.com/datasets/andrewcxjin/whoop-dataset)
- [Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)

### Engineered Features
The following features were programmatically created to enrich the dataset:
- **`systolic_bp` & `diastolic_bp`**: Extracted from a single blood pressure string.
- **`Step_count`, `drinks_per_week`, `stress_rating_1_to_10`**: Synthetically generated to simulate real-world data.
- **`Activity_Level`**: Categorized as 'Sedentary', 'Moderately Active', or 'Active' based on `Step_count`.
- **`High_Stress`**: A binary flag triggered if the stress rating is above 7.
- **`Poor_Sleep`**: A binary flag triggered if sleep is less than 7 hours.
- **`Stress_x_Poor_Sleep`**: An interaction feature combining high stress and poor sleep.
- **`Binge_Flag`**: A binary flag indicating potential binge drinking based on gender and weekly drink count.

## Model Training

The prediction models are `XGBoost Classifiers`. Three separate models were trained to predict `diabetes`, `hypertension`, and `heart_disease` respectively.

The training process is handled by the `create_models.py` script. It loads the `final.csv` dataset, defines the feature and target columns, and then trains and saves each model. The models are configured with `use_label_encoder=False` and `eval_metric='logloss'`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Preygle/welldoc-hackathon.git
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Gemini API Key:**
    Create a `.env` file in the root directory of the project and add the following line:
    ```
    GEMINI_API_KEY="YOUR_API_KEY"
    ```
    Replace `"YOUR_API_KEY"` with your actual Gemini API key.

## Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start the local Streamlit server and open the application in your web browser.
