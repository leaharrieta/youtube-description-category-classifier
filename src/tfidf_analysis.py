# FILE: tfidf_analay
# AUTHOR: Leah Arrieta
# EMAIL: leah.arrieta00@gmail.com
# DATE: 1/6/26
# DESCRIPTION: This file analyzes category level text data using TF-IDF by grouping 
#               descriptions by category, transforming text into numerical feature vectors, 
#               and then identifying the most important category specific terms. This gives 
#               insight to feature relevance and model interpretability.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # To extract in a format supported by ML (text->numerical)

INPUT_FILE = "data/cleaned_dataset.csv"

def main():
    # Load cleaned dataset
    df = pd.read_csv(INPUT_FILE)
    print("Total rows: ", len(df))
    # Get the unique values from the Category coloumn
    print("Categories: ", df["Category"].unique())

    # Valid strings only
    df = df.dropna(subset=["Description"])

    # Make a series for each category, saving only its corresponding description.
    # Also join each category description into one string with a space in between
    category_doc = df.groupby("Category")["Description"].apply(" ".join)

    # Construct an object that keeps a max of 5000 words and removes english stop words
    vectorizer = TfidfVectorizer(max_features = 5000, stop_words = "english")
    # Learn from the vocab data, compute TF-IDF scores produce a matrix
    tfidf_matrix = vectorizer.fit_transform(category_doc)
    # Return words that correspond to TF-IDF coloumns
    feature_names = vectorizer.get_feature_names_out()

    print("\nTop TF-IDF words per category:")
    for index, category in enumerate(category_doc.index):
        # Make row a list-like array of TF-IDF scores for a category
        row = tfidf_matrix[index].toarray()[0]
        # Get the 10 most important words (largest -> smallest)
        top_indices = row.argsort()[-10:][::-1]
        print(f"\nCategory: {category}")

        for i in top_indices:
            # Print each categories most distinctive word and its TF-IDF score
            print(f"{feature_names[i]:15} {row[i]:.4f}")

if __name__ == "__main__":
    main()