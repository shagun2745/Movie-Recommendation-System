import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Load dataset
movies = pd.read_csv("movies.csv")
movies = movies[['original_title', 'overview']]
movies = movies.head(3000)
movies['overview'] = movies['overview'].fillna('')
movies['original_title_lower'] = movies['original_title'].str.lower()

# Convert text to vectors
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['overview'])

# Compute similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend():
    movie = entry.get().lower()

    matches = movies[movies['original_title_lower'].str.contains(movie)]

    if matches.empty:
        messagebox.showinfo("Result", "Movie not found!")
        return

    index = matches.index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Recommended Movies:\n\n")

    for i in movies_list[1:6]:
        result_text.insert(tk.END, "- " + movies.iloc[i[0]].original_title + "\n")

# GUI setup
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x400")

# Title
title_label = tk.Label(root, text="Movie Recommender", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Input field
entry = tk.Entry(root, width=40)
entry.pack(pady=10)

# Button
btn = tk.Button(root, text="Recommend", command=recommend)
btn.pack(pady=10)

# Output box
result_text = tk.Text(root, height=12, width=50)
result_text.pack(pady=10)

# Run app
root.mainloop()