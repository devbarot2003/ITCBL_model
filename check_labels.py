import pandas as pd

df = pd.read_csv("data/csic_database.csv")

print("Unique values in classification column:")
print(df["classification"].unique())

print("\nValue counts:")
print(df["classification"].value_counts())