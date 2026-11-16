
import ast
import re
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "tmdb_5000_movies.csv"

def extract_genres(genre_text):
    """Extract genre names from the CSV genre field."""

    if pd.isna(genre_text) or not str(genre_text).strip():
        return ""

    try:
        genres = ast.literal_eval(genre_text)

        return " ".join(
            genre.get("name", "")
            for genre in genres
            if isinstance(genre, dict)
        )

    except (ValueError, SyntaxError, TypeError):
        return ""

def clean_text(text):
    """Clean text and remove English stopwords."""

    if pd.isna(text):
        text = ""

    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)

def load_and_preprocess():
    """Load dataset and prepare cleaned movie descriptions."""

    df = pd.read_csv(DATASET_PATH)

    required_columns = ["title", "overview", "genres"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    # Handle missing values
    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("")

    # Extract genre names
    df["genre_names"] = df["genres"].apply(extract_genres)

    # Combine overview and genres
    df["combined_text"] = (
        df["overview"] + " " + df["genre_names"]
    )

    # Create clean text
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Remove records without a movie title or usable text
    df = df[
        (df["title"].str.strip() != "") &
        (df["clean_text"].str.strip() != "")
    ].copy()

    # Remove duplicate movie titles
    df = df.drop_duplicates(
        subset=["title"],
        keep="first"
    ).reset_index(drop=True)

    return df

def create_tfidf(df):
    """Convert cleaned movie descriptions into TF-IDF vectors."""

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )

    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

    return vectorizer, tfidf_matrix

def recommend_movies(item_name, df, tfidf_matrix, top_n=5):
    """Return top N movies similar to the selected movie."""

    matching_indices = df.index[
        df["title"].str.casefold() == item_name.strip().casefold()
    ].tolist()

    if not matching_indices:
        return pd.DataFrame()

    movie_index = matching_indices[0]

    # Calculate similarity between the selected movie and all movies
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movies by similarity
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected movie itself
    similar_indices = [
        index for index in similar_indices
        if index != movie_index
    ]

    # Select top N movies
    top_indices = similar_indices[:top_n]

    recommendations = df.iloc[top_indices][
        ["title", "genre_names", "overview"]
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in top_indices
    ]

    return recommendations.reset_index(drop=True)