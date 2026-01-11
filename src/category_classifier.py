import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "data/cleaned_dataset.csv"

def main():
    df = pd.read_csv(INPUT_FILE)
    # Drop empty rows
    df = df.dropna(subset = ["Description", "Category"])

    print("Total rows: ", len(df))

    X = df["Description"]   # Input features (what the model learns from)
    y = df["Category"]      # Labels (what the model is trying to predict)

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.2,
        random_state = 42,
        stratify = y    # For balanced category proportions
    )

    print("Train size: ", len(X_train))
    print("Test size: ", len(X_test))

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features = 5000, stop_words = "english")
    # Learn vocab from training data + convert text to numbers
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Classifier
    model = LogisticRegression(max_iter = 1000) # Higher chance to find a good solution
    # Make the model map the numerical descriptions to the correct categories
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    # Compare predictions to the true answers
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    # To spot class imbalance
    print("Training category counts: ")
    print(y_train.value_counts())

if __name__ == "__main__":
    main()