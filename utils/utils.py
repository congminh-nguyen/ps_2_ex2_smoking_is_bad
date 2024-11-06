import pandas as pd

# Clean the data
def data_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by converting the columns to the appropriate data types.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: Cleaned DataFrame with appropriate data types.
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Dataset contains missing values")
        
    # Change smoker and outcome to binary
    df['smoker'] = df['smoker'].map({"Yes": 1, "No": 0})
    df['outcome'] = df['outcome'].map({"Alive": 1, "Dead": 0})

    # Check if mapping created any NaN values
    if df['smoker'].isnull().any():
        print("Warning: Invalid values in 'smoker' column")
    if df['outcome'].isnull().any():
        print("Warning: Invalid values in 'outcome' column")

    # Change age, outcome, smoker to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')
    df['smoker'] = pd.to_numeric(df['smoker'], errors='coerce')

    # Validate age range
    if (df['age'] < 0).any() or (df['age'] > 120).any():
        print("Warning: Age values outside reasonable range detected")

    # Change gender to categorical
    df['gender'] = df['gender'].astype('category')

    return df
