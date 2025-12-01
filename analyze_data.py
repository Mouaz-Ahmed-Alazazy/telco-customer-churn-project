import pandas as pd

try:
    df = pd.read_csv('train.csv')
    print("Dataset Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    
    if 'Churn' in df.columns:
        print("\nChurn Distribution:\n", df['Churn'].value_counts(normalize=True))
    
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())

except Exception as e:
    print(f"Error: {e}")
