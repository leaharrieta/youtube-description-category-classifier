import pandas as pd

df = pd.read_csv(
    "data/youtube_video_dataset.csv",
    encoding="utf-8",
    engine="python"
)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample rows:\n")
for i in range(3):
    print("TITLE:")
    print(df.iloc[i]["Title"])
    print("\nDESCRIPTION (first 300 chars):")
    print(df.iloc[i]["Description"][:300])
    print("\n" + "-"*50 + "\n")
