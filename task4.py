
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from recommender import load_and_preprocess, create_tfidf

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = load_and_preprocess()

# Create TF-IDF matrix
vectorizer, tfidf_matrix = create_tfidf(df)

# Compute similarity between all movies
similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

print("COSINE SIMILARITY COMPLETED")
print("---------------------------")

print("\nNumber of Movies:")
print(len(df))

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)

print("\nSimilarity Between First Two Movies:")
if len(df) > 1:
    print(similarity_matrix[0, 1])

# Save sparse similarity matrix
with open("outputs/similarity_matrix.pkl", "wb") as file:
    pickle.dump(similarity_matrix, file)

print("\nSimilarity matrix saved successfully!")