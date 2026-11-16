
import os
from recommender import load_and_preprocess

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load and preprocess dataset
df = load_and_preprocess()

print("PREPROCESSING COMPLETED")
print("-----------------------")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Cleaned Movies:")
print(
    df[
        ["title", "genre_names", "clean_text"]
    ].head()
)

print("\nMissing Values in Clean Text:")
print(df["clean_text"].isnull().sum())

# Save cleaned dataset
df.to_csv(
    "outputs/movies_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")