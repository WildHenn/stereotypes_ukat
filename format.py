import pdfplumber
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Ensure NLTK data is downloaded and set up correctly
nltk.data.path.append('/home/codespace/nltk_data')
nltk.download('punkt')

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = 'd6f8a2622d39e9d89ff0dfef1a80ad877f4ee9e3'

# Import the SDK and the client module
from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client

# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Function to extract text from a PDF
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Function to extract date from the text
def extract_date_from_text(text):
    date_patterns = [
        r'\b\d{2}\.\d{2}\.\d{4}\b',  # Matches dd.mm.yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',    # Matches dd/mm/yyyy
        r'\b\d{4}-\d{2}-\d{2}\b',    # Matches yyyy-mm-dd
        r'\b\w+\s\d{1,2},\s\d{4}\b'  # Matches Month dd, yyyy
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return None

# Function to convert date format to yyyy-mm-dd (if necessary)
def convert_date_format(date_string):
    formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%B %d, %Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None

# Function to tokenize, lowercase, and clean the text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation and single characters
    tokens = [token for token in tokens if token not in string.punctuation and len(token) > 1]
    return ' '.join(tokens)  # Return as a string of words, since TF-IDF expects strings, not tokens

# Function to process the PDF and extract text and date
def process_pdf(pdf_path):
    # Step 1: Extract text from the PDF
    text = extract_pdf_text(pdf_path)
    
    # Define the pattern for splitting articles
    article_marker_pattern = r'\d{2}\.\d{2}\.\d{4}\sSeite\s\d+'
    
    # Split the text into articles based on the marker pattern
    articles = re.split(article_marker_pattern, text)
    
    results = []
    for article in articles[1:]:  # Skip the first split part if it is before the first marker
        # Extract and format the date for each article
        raw_date = extract_date_from_text(article)
        formatted_date = convert_date_format(raw_date) if raw_date else 'No date found'
        
        # Preprocess the article text (lowercase, tokenize, clean)
        preprocessed_text = preprocess_text(article.strip())
        
        results.append((formatted_date, preprocessed_text))
    
    return results

# Example usage with the path to your PDF in the repository
pdf_path = 'sample_articles.pdf'  # Ensure this path is correct relative to your script
articles_data = process_pdf(pdf_path)

# Extract only the preprocessed article texts for TF-IDF
article_texts = [text for date, text in articles_data]

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(article_texts)

# Get the feature names (i.e., the words) and print the TF-IDF scores for each document
feature_names = vectorizer.get_feature_names_out()

# Output the results
for i, article in enumerate(articles_data):
    print(f"Article {i+1} Date: {article[0]}")
    print(f"TF-IDF for Article {i+1}:")
    
    # Get the TF-IDF values for each word in the article
    tfidf_scores = tfidf_matrix[i].T.todense()
    
    # Print the words with their respective TF-IDF scores
    word_scores = dict(zip(feature_names, tfidf_scores.tolist()))
    
    # Sort words by TF-IDF score in descending order
    sorted_word_scores = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10 words with the highest TF-IDF scores
    for word, score in sorted_word_scores[:10]:
        print(f"{word}: {score}")
    
    print('-' * 40)  # Separator between articles

    import pandas as pd

# Function to save extracted data to a CSV file
def save_to_csv(articles_data, output_filename="articles.csv"):
    # Convert the extracted data into a Pandas DataFrame
    df = pd.DataFrame(articles_data, columns=["Date", "Text"])

    # Save to CSV
    df.to_csv(output_filename, index=False, encoding="utf-8")
    print(f"CSV file saved as {output_filename}")

# Example usage: Convert extracted articles to CSV
csv_filename = "articles.csv"
save_to_csv(articles_data, csv_filename)