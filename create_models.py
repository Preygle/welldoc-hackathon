import pandas as pd
import xgboost as xgb
import pickle

# Load the dataset
df = pd.read_csv("final.csv")

# Define features (X) and targets (y)
features = [
    'gender', 'age', 'bmi', 'glucose level', 'heart rate', 'sleep hours', 'sugar level', 'cholesterol',
    'systolic_bp', 'diastolic_bp', 'smoking_status', 'Step_count', 'drinks_per_week',
    'stress_rating_1_to_10', 'Activity_Level', 'Alcohol_Use', 'Binge_Flag', 'High_Stress',
    'Poor_Sleep', 'Stress_x_Poor_Sleep'
]

targets = ['diabetes', 'hypertension', 'heart_disease']

X = df[features]

for target in targets:
    print(f"Training model for {target}...")
    y = df[target]

    # Create and train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    # Save the model to a .pkl file
    with open(f"{target}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model for {target} saved as {target}_model.pkl")

print("All models have been created successfully.")
