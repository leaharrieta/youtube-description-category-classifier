# FILE: text_stats.py
# AUTHOR: Leah Arrieta
# EMAIL: leah.arrieta00@gmail.com
# DATE: 12/29/25
# DESCRIPTION: This file performs exploratory text analysis on the cleaned dataset. It tokenizes 
#               descriptions, removes stopwords, computes word counts, vocabulary size, 
#               and identifies the most frequent words across the dataset to better 
#               understand the text structure, content distribution, and inform future decisions.

import pandas as pd
from collections import Counter

INPUT_FILE = "data/cleaned_dataset.csv"

STOPWORDS = {
    "the", "and", "to", "of", "in", "a", "is", "on", "for", "with",
    "that", "this", "it", "you", "i", "my", "we", "our", "your",
    "are", "be", "as", "at", "by", "from", "or", "an", "&", ":", "-", 
    "twitter:", "facebook:", "instagram:", "►", "|", "*"
}

# def tokenize() - A function that ignores non-string characters and returns
# text in lowercase form and split into tokens
def tokenize(text):
    if not isinstance(text, str):
        return []
        
    tokens = text.lower().split()
    return [t for t in tokens if t not in STOPWORDS]

def main():
    df = pd.read_csv(INPUT_FILE)

    print("Total rows: ", df.shape[0])

    # Tokenize descriptions
    df["tokens"] = df["Description"].apply(tokenize)

    # Drop rows with no tokens
    df = df[df["tokens"].map(len) > 0]

    # Get the word count of each description
    df["word_count"] = df["tokens"].apply(len)

    print("Word Count Summaries: ")
    print(df["word_count"].describe())

    # Tokenize the entire dataset into a single list
    all_tokens = []
    for token_list in df["tokens"]:
        for token in token_list:
            all_tokens.append(token)

    # Build bigrams
    all_bigrams = []

    for tokens in df["tokens"]:
        # Form adjacent word pairs
        bigrams = zip(tokens, tokens[1:])
        all_bigrams.extend(bigrams)

    # Vocabulary size, set of all unique words used anywhere in the dataset
    vocab = set(all_tokens)
    print("\nVocabulary size:", len(vocab))

    #
    print("\nCategory-based top words:")

    # Split df into smaller data frames, per unique "Category" value
    for category, group in df.groupby("Category"):
        tokens = [] # List to hold category tokens
        # For each row in that category, add its token to the list
        for row_tokens in group["tokens"]:
            tokens.extend(row_tokens)

        # Count word frequencies
        counter = Counter(tokens)

        # Print ten most common words in each category 
        print(f"\nCategory: {category}")
        for word, count in counter.most_common(10):
            print(f"{word:15} {count}")

    # Proudce a frequency map to show the most common words
    counter = Counter(all_tokens)
    print("\nTop 20 most common words:")
    for word, count in counter.most_common(20):
        print(f"{word:15}{count}")

    # Count bigram frequency
    bigram_counts = Counter(all_bigrams)
    print("\nTop 20 most common bigrams:")
    for (w1, w2), count in bigram_counts.most_common(20):
        print(f"{w1:15} {w2:15} {count:5}")

if __name__ == "__main__":
    main()
