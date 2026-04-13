import pandas as pd
df = pd.read_csv("data/csic_database.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())