# Assignment 20: Building and Deploying a Recommendation System

## Project Overview

This project implements a content-based movie recommendation system using TF-IDF vectorization and cosine similarity.

Users can select a movie and receive recommendations for similar movies through a Streamlit web application.

## Dataset

- **Dataset Name:** TMDB 5000 Movie Dataset
- **Source:** Kaggle
- **Dataset Link:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- **Input File:** tmdb_5000_movies.csv
- **Item Identifier:** title
- **Text Features:** overview and genres

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Git
- GitHub
- Render

## Tasks Completed

1. Dataset Loading and Understanding
2. Text Preprocessing
3. TF-IDF Vectorization
4. Cosine Similarity Computation
5. Movie Recommendation Function
6. Streamlit User Interface
7. Git and GitHub Setup
8. Render Deployment
9. Final Application Validation

## Recommendation Method

### Text Preprocessing

- Converted text to lowercase.
- Removed punctuation and special characters.
- Removed English stopwords.
- Handled missing values.
- Combined movie overview and genres.

### TF-IDF Vectorization

Used TfidfVectorizer with the following parameters:

- max_features = 5000
- ngram_range = (1, 2)
- min_df = 2
- max_df = 0.85

### Similarity Computation

Cosine similarity was used to compare movie descriptions and identify similar movies.

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## GitHub Repository

https://github.com/PyDevSachin/Assignment_20_Movie_Recommendation_System

## Deployed Application

https://assignment-20-movie-recommendation-system.onrender.com

## Conclusion

A content-based movie recommendation system was developed using TF-IDF and cosine similarity.

A simple Streamlit interface was created to allow users to select movies and receive recommendations.

The project was uploaded to GitHub using Git and deployed on Render.