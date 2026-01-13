# FILE: inspect_data.py
# AUTHOR: Leah Arrieta
# EMAIL: leah.arrieta00@gmail.com
# DATE: 12/22/25
# DESCRIPTION: This file serves as the initial inspection of the raw YouTube video dataset.
#               The CSV file is loaded into a pandas DataFrame then, the dataset shape and
#               column names are printed. Sample titles and descriptions are also printed to provide an
#               overview of the data structure, content, and formatting before any cleaning
#               or analysis is performed.

# import pandas library and remane to pd
import pandas as pd

# Read the csv file and convert into a data frame
file = "data/youtube_video_dataset.csv"

try:
    df = pd.read_csv(
        file,
        encoding = "utf-8", # To handle global text (emojis, special chars)
        engine = "python")  # To handle multiline text
    
    print("Dataframe successfully created from file.")
    print("Dataset Shape:", df.shape)   # makes (rows, cols)
    print("\nColoums:")
    print(df.columns)

    # Print the first three data entries
    print("\nSample Rows:")
    for i in range(3):
        print("\nTITLE:")
        print(df.iloc[i]["Title"])  # iloc (integer location) give row at location i
        print("\nDESCRIPTION (first 300 chars):")
        print(df.iloc[i]["Description"][:300])
        print("\n" + "-" * 50 + "\n")   # For readability

except FileNotFoundError:
    print(f"File '{file}' not found")
except Exception as e:
    print(f"An error occured {e}.")