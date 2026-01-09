import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/cleaned_dataset.csv"

def main():
    df = pd.read_csv(INPUT_FILE)
    # Drop empty rows
    df = df.dropna(subset = ["Description", "Category"])

    print("Total rows: ", len(df))

    X = df["Description"]   # Input features (to learn from)
    y = df["Category"]      # Labels (correct category for each description)

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.2,
        random_state = 42,
        stratify = y    # For balanced category proportions
    )

    print("Train size: ", len(X_train))
    print("Test size: ", len(X_test))

    # To spot class imbalance
    print("Training category counts: ")
    print(y_train.value_counts())

if __name__ == "__main__":
    main()