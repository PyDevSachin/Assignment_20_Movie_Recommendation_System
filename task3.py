
import os
import pickle

from recommender import load_and_preprocess, create_tfidf

os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = load_and_preprocess()

# Create TF-IDF vectors
vectorizer, tfidf_matrix = create_tfidf(df)

print("TF-IDF VECTORIZATION COMPLETED")
print("------------------------------")

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

print("\nVocabulary Size:")
print(len(vectorizer.get_feature_names_out()))

print("\nFirst 20 Vocabulary Words:")
print(vectorizer.get_feature_names_out()[:20])

# Save vectorizer and TF-IDF matrix
with open("outputs/tfidf_data.pkl", "wb") as file:
    pickle.dump(
        (vectorizer, tfidf_matrix),
        file
    )

print("\nTF-IDF data saved successfully!")