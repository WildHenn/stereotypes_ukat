import pandas as pd
import re

# Load dataset (change the path if necessary)
file_path = "jean_data_apa.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Print the column names to confirm structurex
print("Columns in dataset:", df.columns)

# Check if the right column for filtering exists
if "text" not in df.columns:
    raise KeyError("Column 'text' not found. Please check the column names.")

# Define keywords related to Africa and racism/discrimination
africa_keywords = [
    "Schwarze", "Afrikaner", "Afrikanische Wurzeln", "Afrikanischer Hintergrund", 
    "Afrikanische Herkunft", "Afro", "Dunkelhäutig", "Schwarzafrika", "Sub-Sahara Afrika",
    "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Kamerun", "Kap Verde", 
    "Zentralafrikanische Republik", "Tschad", "Komoren", "Kongo", "Demokratische Republik Kongo", 
    "Dschibuti", "Äquatorialguinea", "Eritrea", "Swasiland", "Äthiopien", "Gabun", "Gambia", 
    "Ghana", "Guinea", "Guinea-Bissau", "Elfenbeinküste", "Kenia", "Lesotho", "Liberia", 
    "Madagaskar", "Malawi", "Mali", "Mauretanien", "Mauritius", "Mosambik", "Namibia", 
    "Niger", "Nigeria", "Ruanda", "São Tomé und Príncipe", "Senegal", "Seychellen", 
    "Sierra Leone", "Somalia", "Südafrika", "Südsudan", "Sudan", "Tansania", "Togo", 
    "Uganda", "Sambia", "Simbabwe"
]

discrimination_keywords = [
    "Rassismus", "Diskriminierung", "Vorurteile", "Stereotyp", "Fremdenfeindlichkeit",
    "Benachteiligung", "Hassverbrechen", "Racial Profiling", "Kolonialismus", "Struktureller Rassismus",
    "Antischwarzer Rassismus", "Weiße Vorherrschaft", "Systemischer Rassismus", "Rassistische Gewalt"
]

# Convert lists into regex search patterns
africa_pattern = "|".join(africa_keywords)
discrimination_pattern = "|".join(discrimination_keywords)

# Define exclusion keywords for irrelevant contexts (like fashion or events)
exclude_keywords = [
    "Mode", "Kleider", "Veranstaltung", "Event", "Schwarzenegger"
]

# Create a regex pattern for exclusion
exclude_pattern = "|".join(exclude_keywords)

# Define a function to filter out event-like content
def filter_event_articles_content(articles_content):
    """
    Filters out articles content that seem to be event listings based on common patterns (e.g., phone numbers, URLs, date formats).
    
    :param articles_content: List of article content (strings)
    :return: List of article content that don't match event-like patterns
    """
    event_patterns = [
        r'https?://[^\s]+',  # URL pattern
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # Date pattern (DD/MM/YYYY or DD-MM-YYYY)
        r'\d{1,2}:\d{1,2}',  # Time pattern (HH:MM)
        r'\d{10}',  # Phone number pattern (10 digits)
        r'\b\d{4,5}\b',  # Postal codes (usually 4 or 5 digits in Austria)
        r'\b[A-Za-z]+\s[A-Za-z]+\b',  # Simple name pattern (could catch addresses like 'Wiener Stadthalle')
    ]
    
    filtered_articles = []
    for content in articles_content:
        # Ensure content is treated as a string, defaulting to an empty string if it's NaN or not a string
        content = str(content) if isinstance(content, str) else str(content)
        if not any(re.search(pattern, content) for pattern in event_patterns):
            filtered_articles.append(content)
    
    return filtered_articles

# Filter the content of articles to exclude event-like content
filtered_content = filter_event_articles_content(df["text"])

# Apply filtering: articles must mention at least one Africa-related AND one discrimination-related term
# and must not include any of the exclusion terms
filtered_df = df[
    df["text"].isin(filtered_content) &  # Filter the articles that passed event exclusion
    df["text"].str.contains(africa_pattern, case=False, na=False) &
    df["text"].str.contains(discrimination_pattern, case=False, na=False) &
    ~df["text"].str.contains(exclude_pattern, case=False, na=False)
]

# Remove duplicate articles based on 'text' column
filtered_df = filtered_df.drop_duplicates(subset=["text"])

# Save the filtered dataset
filtered_df.to_csv("filtered_articles3.csv", index=False, encoding="utf-8")

# Display final number of articles
print(f"Number of filtered articles: {len(filtered_df)}")

# Take a random sample of 50 articles for manual review
sample_df = filtered_df.sample(50, random_state=42)

# Show the sampled articles (headline & text)
print(sample_df[["headline", "text"]])

# Save the sample to a separate CSV for manual inspection
sample_df.to_csv("sample_articles3.csv", index=False, encoding="utf-8")
