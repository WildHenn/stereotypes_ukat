import pandas as pd

# Load your filtered dataset
df = pd.read_csv("filtered_articles_expanded.csv", encoding="utf-8")

# Sample 100 random articles
df_sample = df.sample(n=100)

# Save the sample dataset to a new CSV file for categorization in Label Studio
df_sample.to_csv("sample_for_labeling.csv", encoding="utf-8", index=False)

print(f"100 sample articles have been saved to 'sample_for_labeling.csv'with {len(df)} articles.")
