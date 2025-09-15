'''
This script adds new features to the fabricated_diabetes_dataset.csv based on gem.txt.
'''
import pandas as pd
import numpy as np
import re

def add_features():
    '''
    Loads the dataset, adds new features, and saves it to a new CSV file.
    '''
    try:
        df = pd.read_csv('fabricated_diabetes_dataset.csv')
    except FileNotFoundError:
        print("Error: 'fabricated_diabetes_dataset.csv' not found. Make sure the file is in the correct directory.")
        return

    # --- Pre-processing existing columns ---
    if 'BP' in df.columns and df['BP'].dtype == 'object':
        df[['systolic_bp', 'diastolic_bp']] = df['BP'].str.split('/', expand=True)
        df['systolic_bp'] = pd.to_numeric(df['systolic_bp'], errors='coerce')
        df['diastolic_bp'] = pd.to_numeric(df['diastolic_bp'], errors='coerce')
        df.drop('BP', axis=1, inplace=True)

    if 'Medical History' in df.columns and df['Medical History'].dtype == 'object':
        df['smoking_status'] = df['Medical History'].apply(
            lambda x: re.search(r"Smoking: ([\w\s]+?),", str(x)).group(1).strip() if re.search(r"Smoking: ([\w\s]+?),", str(x)) else 'No Info'
        )

    # --- Generate Pseudo-Columns for missing data ---
    np.random.seed(42)
    df['Step_count'] = np.random.randint(2000, 15001, size=len(df))
    df['drinks_per_week'] = np.random.geometric(p=0.2, size=len(df)) - 1
    df.loc[df['drinks_per_week'] > 20, 'drinks_per_week'] = 20
    df['stress_rating_1_to_10'] = np.random.randint(1, 11, size=len(df))

    # --- Feature Engineering ---
    def activity_level(steps):
        if steps < 5000:
            return 'Sedentary'
        elif 5000 <= steps < 10000:
            return 'Moderately Active'
        else:
            return 'Active'

    df['Activity_Level'] = df['Step_count'].apply(activity_level)

    df['Alcohol_Use'] = (df['drinks_per_week'] > 0).astype(int)

    df['Binge_Flag'] = 0
    df.loc[(df['gender'] == 'Female') & (df['drinks_per_week'] >= 4), 'Binge_Flag'] = 1
    df.loc[(df['gender'] == 'Male') & (df['drinks_per_week'] >= 5), 'Binge_Flag'] = 1
    if 'Other' in df['gender'].unique():
        df.loc[(df['gender'] == 'Other') & (df['drinks_per_week'] >= 5), 'Binge_Flag'] = 1

    df['High_Stress'] = (df['stress_rating_1_to_10'] > 6).astype(int)

    df['Poor_Sleep'] = (df['sleep hours'] < 7).astype(int)

    df['Stress_x_Poor_Sleep'] = df['High_Stress'] * df['Poor_Sleep']

    # --- Save the new dataframe ---
    output_filename = 'fabricated_diabetes_dataset_with_features.csv'
    df.to_csv(output_filename, index=False)

    print(f"Successfully added features and saved the new dataset as '{output_filename}'")

if __name__ == "__main__":
    add_features()
