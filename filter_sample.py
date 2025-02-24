import pandas as pd

# Load your filtered dataset
df = pd.read_csv("data/filtered_articles_expanded.csv", encoding="utf-8")

# Sample 800 random articles
df_sample = df.sample(n=800)

# Save the sample dataset to a new CSV file for categorization in Label Studio
df_sample.to_csv("sample_for_labeling_2.csv", encoding="utf-8", index=False)

print(f"800 sample articles have been saved to 'sample_for_labeling_2.csv'with {len(df)} articles.")
