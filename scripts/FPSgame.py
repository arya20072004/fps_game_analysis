import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Extract: Load raw match data."""
    return pd.read_csv("VRLMaster_cleaned.csv")

def clean_data(df):
    """Clean: Handle missing values, rename columns."""
    df.columns = [col.replace('%', 'Percent').replace(' ', '_') for col in df.columns]
    df['CLPercent'].fillna(df['CLPercent'].median(), inplace=True)
    return df

def transform_data(df):
    """Transform: Create new features, normalize values."""
    # Convert percentages
    percent_cols = ['KAST', 'HSPercent', 'CLPercent']
    df[percent_cols] = df[percent_cols] / 100

    # Feature engineering: Kill Participation
    df['Kill_Participation'] = (df['K'] + df['A']) / df['Rnd']

    # Normalize key metrics
    scaler = MinMaxScaler()
    df[['ACS', 'ADR', 'K_D']] = scaler.fit_transform(df[['ACS', 'ADR', 'K_D']])
    
    return df

def save_cleaned_data(df, output_path):
    """Load: Save cleaned data to file/database."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    raw_data_path = "FPS_Game_RawData.csv"
    cleaned_data_path = "FPS_Game_Cleaned.csv"

    df_raw = load_data(raw_data_path)
    df_cleaned = clean_data(df_raw)
    df_transformed = transform_data(df_cleaned)
    save_cleaned_data(df_transformed, cleaned_data_path)
