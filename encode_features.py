'''
This script encodes categorical columns to integers for the diabetes dataset.
'''
import pandas as pd

def encode_features():
    '''
    Loads the dataset, encodes specified columns, and saves it back to the same file.
    '''
    file_path = 'fabricated_diabetes_dataset_with_features.csv'

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the file exists in the directory.")
        return

    # Encode "Activity_Level" to integers
    if 'Activity_Level' in df.columns:
        activity_mapping = {'Sedentary': 0, 'Moderately Active': 1, 'Active': 2}
        df['Activity_Level'] = df['Activity_Level'].map(activity_mapping)
        print("Successfully encoded the 'Activity_Level' column to integers.")

    # Encode "gender" to integers
    if 'gender' in df.columns:
        gender_mapping = {'Male': 1, 'Female': 0}
        # Check for other gender values before mapping
        other_genders = df[~df['gender'].isin(gender_mapping.keys())]
        if not other_genders.empty:
            print(f"Found other gender values: {other_genders['gender'].unique()}. They will be mapped to NaN.")
        df['gender'] = df['gender'].map(gender_mapping)
        print("Successfully encoded the 'gender' column to integers (Male: 1, Female: 0).")

    # Save the modified DataFrame, overwriting the old file
    df.to_csv(file_path, index=False)
    print(f"Changes have been saved to '{file_path}'.")

if __name__ == "__main__":
    encode_features()
