import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Script started")


# Load dataset
df = pd.read_csv("sample_classifier.csv", encoding="utf-8")

print("Dataset loaded successfully")
print(df.head())


# Check for missing values
df.dropna(subset=["text", "sentiment"], inplace=True)

# Normalize labels
df["sentiment"] = df["sentiment"].str.strip().str.lower()

# Encode labels
label_mapping = {"irrelevant": 0, "relevant": 1}
df["sentiment"] = df["sentiment"].map(label_mapping)
print("Label distribution after mapping:")
print(df["sentiment"].value_counts())  # Ensure both 0s and 1s exist

# Check if both classes exist
if df["sentiment"].nunique() < 2:
    raise ValueError("Dataset must contain at least two classes for classification.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])
print("Training label distribution:")
print(y_train.value_counts())  # Ensure both 0s and 1s exist

# Vectorize text data
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF transformation complete")
print(f"X_train shape: {X_train_tfidf.shape}")
print(f"X_test shape: {X_test_tfidf.shape}")

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

print("Model training complete")

# Predict
y_pred = clf.predict(X_test_tfidf)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
