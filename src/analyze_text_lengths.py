# FILE: analyze_text_lengths.py
# AUTHOR: Leah Arrieta
# EMAIL: leah.arrieta00@gmail.com
# DATE: 12/23/25
# DESCRIPTION: This file examines text length characteristics of YouTube video titles and descriptions. 
#               It computes summary statistics, percentiles, and identifies
#               extremely long descriptions to inform preprocessing decisions such as
#               maximum description length truncation.

import pandas as pd

file = "data/youtube_video_dataset.csv"

def main():
    df = pd.read_csv(
        file,
        encoding = "utf-8",
        engine = "python"
        )
    print("Total entries: ", df.shape[0])   # Number of rows

    # Make a new coloum in df for num of chars in each title 
    # and another for num of chars in each description
    df["title_length"] = df["Title"].str.len()  # Process as a string and get length
    df["des_length"] = df["Description"].str.len()

    # Get summary stats on the now numeric data
    print(df["title_length"].describe())
    print(df["des_length"].describe())

    # Compute cut points to later choose a cut off
    percentiles = df["des_length"].quantile([0.90, 0.95, 0.99])

    # How many extreme longs exist?
    long_2000 = (df["des_length"] > 2000).sum() # Returns total num of true
    long_3000 = (df["des_length"] > 3000).sum()

    # Get examples of long descriptions (only 3 largest)
    long_example = df.sort_values("des_length", ascending= False).head(3)

    # Iterate through each row of long_example and print
    for _, row in long_example.iterrows():
        print("\nTitle:", row["Title"])
        print("Description length: ", row["des_length"])
        print(row["Description"][:300], "...")

if __name__ == "__main__":
    main()