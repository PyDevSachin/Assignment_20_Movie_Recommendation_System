
import pandas as pd

# Load the Kaggle movie dataset
df = pd.read_csv("tmdb_5000_movies.csv")

print("DATASET INFORMATION")
print("-------------------")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Identify important columns
print("\nColumns Used for Recommendation:")
print("Movie identifier: title")
print("Text features: overview and genres")

print("\nTotal Movies:", len(df))