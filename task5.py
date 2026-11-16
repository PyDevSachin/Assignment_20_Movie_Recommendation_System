
from recommender import (
    load_and_preprocess,
    create_tfidf,
    recommend_movies
)

# Load and prepare movie data
df = load_and_preprocess()

# Create TF-IDF matrix
vectorizer, tfidf_matrix = create_tfidf(df)

# Test with 3 different movies
test_movies = [
    "Avatar",
    "The Dark Knight",
    "Toy Story"
]

for movie in test_movies:

    print("\n" + "=" * 55)
    print("SELECTED MOVIE:", movie)
    print("=" * 55)

    recommendations = recommend_movies(
        movie,
        df,
        tfidf_matrix,
        top_n=5
    )

    if recommendations.empty:
        print("Movie not found in the dataset.")
    else:
        print(recommendations.to_string(index=False))