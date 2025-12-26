import pandas as pd
import re

INPUT_FILE = "data/youtube_video_dataset.csv"
OUTPUT_FILE = "data/cleaned_dataset.csv"
MAX_LENGTH = 1024

def remove_emoji(text):
    # Check if input is a string
    if not isinstance(text, str):
        return ""
    
    emoji_pattern = re.compile(
        # Create a character class to identify emoji related characters
        "["
        "\U0001F600-\U0001F64F" # Emoticons
        "\U0001F300-\U0001F5FF" # Symbols + Pictographs
        "\U0001F680-\U0001F6FF" # Transport + Map Symbols
        "\U0001F1E0-\U0001F1FF" # Flags
        "]+",                   # One or more times
        # Treat pattern and input as unicode
        flags = re.UNICODE
    )

    # Replace emojis with an empty string
    return emoji_pattern.sub("", text)

def main():
    df = pd.read_csv(
        INPUT_FILE,
        encoding = "utf-8",
        engine = "python"
    )

    print("Original rows: ", df.shape[0])

    # Remove missing values
    df = df.dropna(subset = ["Title", "Description"])
    # Strip whitespace
    df["Title"] = df["Title"].str.strip()
    df["Description"] = df["Description"].str.strip()
    # Remove emojis
    df["Title"] = df["Title"].apply(remove_emoji)
    df["Description"] = df["Description"].apply(remove_emoji)
    # Truncate desctiptions
    df["Description"] = df["Description"].str.slice(0, MAX_LENGTH)
    # Save clean dataset
    df.to_csv(OUTPUT_FILE, index = False)   # Do not add row name index

if __name__ == "__main__":
    main()