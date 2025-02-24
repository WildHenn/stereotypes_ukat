import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the labeled dataset (replace with your actual file path)
df_labeled = pd.read_csv("data/sample_classifier.csv", encoding="utf-8")

# Check the first few rows to ensure the data is loaded correctly
print("Loaded labeled dataset:")
print(df_labeled.head())

# Check the raw label distribution in the 'sentiment' column
print("Raw sentiment values:")
print(df_labeled["sentiment"].value_counts())

# Step 2: Preprocess the text data
# Convert all text to string and handle missing values
df_labeled["text"] = df_labeled["text"].astype(str).fillna("")
# Clean the text: lowercase and remove non-alphanumeric characters
df_labeled["text"] = df_labeled["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x))

# Step 3: Process the labels
# Since the CSV already contains textual labels ("relevant" and "irrelevant"),
# we simply clean them (remove extra whitespace and convert to lowercase)
df_labeled["sentiment"] = df_labeled["sentiment"].astype(str).str.strip().str.lower()

# Print the distribution after cleaning to ensure both classes are present
print("Label distribution after cleaning:")
print(df_labeled["sentiment"].value_counts())

# Encode labels ('relevant' and 'irrelevant')
label_encoder = LabelEncoder()
df_labeled["label_encoded"] = label_encoder.fit_transform(df_labeled["sentiment"])

print("Encoded label classes:", label_encoder.classes_)
print("Encoded label distribution:")
print(pd.Series(df_labeled["label_encoded"]).value_counts())

# Step 4: Split the data into features (X) and labels (y)
X = df_labeled["text"]
y = df_labeled["label_encoded"]

# Use stratified splitting to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set label distribution:")
print(pd.Series(y_train).value_counts())

# Step 5: Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train a Classifier (logistic Regression)
clf = LogisticRegression(class_weight='balanced', random_state=42)  # Use LogisticRegeression
clf.fit(X_train_tfidf, y_train)

# Step 7: Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Label Encoder Classes:", label_encoder.classes_)
print("Data Type of Classes:", type(label_encoder.classes_[0]))

# Fix: Manually define target names instead of using label_encoder.classes_
target_names = ["irrelevant", "relevant"]
target_names = list(map(str, label_encoder.classes_))
print(classification_report(y_test, y_pred, target_names=target_names))

# Step 9: Save the trained model and vectorizer for future use (optional)
joblib.dump(clf, "text_classifier_model_rf.pkl")  # Save the LogisticRegression model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully.")
