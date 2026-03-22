import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Use required columns
movies = movies[['original_title', 'overview']]

# Reduce dataset size (to avoid memory error)
movies = movies.head(3000)

# Fill missing values
movies['overview'] = movies['overview'].fillna('')

# Convert text to lowercase (for better matching)
movies['original_title_lower'] = movies['original_title'].str.lower()

# Convert text to vectors
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['overview'])

# Compute similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    movie = movie.lower()

    # Find closest matching movie
    matches = movies[movies['original_title_lower'].str.contains(movie)]

    if matches.empty:
        print("Movie not found!")
        return

    index = matches.index[0]

    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)

    print("\nRecommended movies:\n")
    for i in movies_list[1:6]:
        print("- " + movies.iloc[i[0]].original_title)

# User input
movie_name = input("Enter movie name: ")
recommend(movie_name)

