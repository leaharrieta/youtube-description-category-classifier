import pandas as pd
from collections import Counter

INPUT_FILE = "data/cleaned_dataset.csv"

# def tokenize() - A function that ignores non-string characters and returns
# text in lowercase form and split into tokens
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()

def main():
    df = pd.read_csv(INPUT_FILE)

    print("Total rows: ", df.shape[0])

    # Tokenize descriptions
    df["tokens"] = df["Description"].apply(tokenize)

    # Get the word count of each description
    df["word_count"] = df["tokens"].apply(len)

    print("Word Count Summaries: ")
    print(df["word_count"].describe())

    # Tokenize the entire dataset into a single list
    all_tokens = []
    for i in df["tokens"]:
        for j in i:
            all_tokens.append(j)


    # Vocabulary size, set of all unique words used anywhere in the dataset
    vocab = set(all_tokens)
    print("\nVocabulary size:", len(vocab))

    # Proudce a frequency map to show the most common words
    counter = Counter(all_tokens)

    print("\nTop 20 most common words:")
    for word, count in counter.most_common(20):
        print(f"{word:15}{count}")

if __name__ == "__main__":
    main()
