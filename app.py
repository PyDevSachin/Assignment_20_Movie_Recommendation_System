
import streamlit as st

from recommender import (
    load_and_preprocess,
    create_tfidf,
    recommend_movies
)

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load and prepare data only once
@st.cache_resource
def load_recommendation_data():

    df = load_and_preprocess()

    vectorizer, tfidf_matrix = create_tfidf(df)

    return df, tfidf_matrix

# Application title
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movies "
    "using content-based filtering."
)

st.divider()

# Load recommendation data
try:
    with st.spinner("Loading movie database..."):
        df, tfidf_matrix = load_recommendation_data()

except Exception as error:
    st.error(f"Unable to load movie data: {error}")
    st.stop()

st.success(f"Successfully loaded {len(df)} movies!")

# Movie selection
movie_list = sorted(df["title"].tolist())

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

# Number of recommendations
top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)

# Recommendation button
if st.button("Get Recommendations", type="primary"):

    recommendations = recommend_movies(
        selected_movie,
        df,
        tfidf_matrix,
        top_n
    )

    if recommendations.empty:
        st.warning("No recommendations found.")

    else:
        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        for index, row in recommendations.iterrows():

            st.markdown(f"### {index + 1}. {row['title']}")

            st.write(
                f"**Genres:** {row['genre_names']}"
            )

            st.write(
                f"**Similarity Score:** "
                f"{row['similarity_score']:.4f}"
            )

            if row["overview"]:
                st.write(row["overview"])
            else:
                st.write("No movie description available.")

            st.divider()

st.caption(
    "Recommendation technique: TF-IDF + Cosine Similarity"
)