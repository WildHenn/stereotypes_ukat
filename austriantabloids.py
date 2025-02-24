import pandas as pd
import re

# Load the CSV file (adjust delimiter if needed, e.g., ";" for German Excel users)
df = pd.read_csv("jean_data_apa.csv", encoding="utf-8", delimiter=",")

# Display first few rows and column names to inspect the data
print(df.head())
print("Columns in dataset:", df.columns)

# 🟢 Step 1: Keep only relevant columns (adjust column names based on your dataset)
columns_to_keep = ["id", "headline", "text", "date", "source"]  # Adjust based on your CSV
df = df[columns_to_keep]

# 🟢 Step 2: Remove duplicates (keeping only unique articles based on headline + text)
df = df.drop_duplicates(subset=["headline", "text"])

# 🟢 Step 3: Remove articles that do not contain certain keywords in title or content
africa_keywords = [
    "Schwarze", "Afrikaner", "Afrikanische Wurzeln", "Afrikanischer Hintergrund", "Afrikanische Herkunft", 
    "Afro", "Dunkelhäutig", "Schwarzafrika", "Sub-Sahara Afrika", "Angola", "Benin", "Botswana", 
    "Burkina Faso", "Burundi", "Kamerun", "Kap Verde", "Zentralafrikanische Republik", "Tschad", "Komoren", 
    "Kongo", "Demokratische Republik Kongo", "Dschibuti", "Äquatorialguinea", "Eritrea", "Swasiland", 
    "Äthiopien", "Gabun", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Elfenbeinküste", "Kenia", "Lesotho", 
    "Liberia", "Madagaskar", "Malawi", "Mali", "Mauretanien", "Mauritius", "Mosambik", "Namibia", "Niger", 
    "Nigeria", "Ruanda", "São Tomé und Príncipe", "Senegal", "Seychellen", "Sierra Leone", "Somalia", 
    "Südafrika", "Südsudan", "Sudan", "Tansania", "Togo", "Uganda", "Sambia", "Simbabwe"
]  # Adjust based on your research focus
discrimination_keywords = ["Rassismus", "Diskriminierung", "Vorurteil", "Stereotyp", "Kolonial", "Benachteiligung", "Hass",
    "Rassistisch", "Fremdenfeindlichkeit", "Xenophobie", "Hautfarbe", "Ethnisch", "Herkunft", "Minderheit",
    "Apartheid", "Menschenrechte", "Antirassismus"]

# Combine both categories into a regex pattern
keyword_pattern = "|".join(africa_keywords)  # Matches African-related terms
discrimination_pattern = "|".join(discrimination_keywords)  # Matches racism-related terms

# Filter articles containing BOTH African terms and discrimination terms
df = df[
    df["text"].str.contains(keyword_pattern, case=False, na=False) |
    df["headline"].str.contains(keyword_pattern, case=False, na=False) |
    df["text"].str.contains(discrimination_pattern, case=False, na=False) |
    df["headline"].str.contains(discrimination_pattern, case=False, na=False)
]

# Step 4: Exclude articles about animals, zoos, and Schwarzenegger
exclude_keywords = ["Tier", "Zoo", "Schwarzenegger", "Zutaten", "schwarzer Peter", "Mode", "schwarzes Kleid", "OEVP", "Schwarzenberg"]
df = df[~df["text"].str.contains("|".join(exclude_keywords), case=False, na=False)]

# 🟢 Step 5: Define a function to filter out event-like content
def is_event_listing(text):
    """
    Checks if an article consists mainly of event listings, theaters, cinemas, webpages, or dates.
    Returns True if it matches these patterns and should be excluded.
    """
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN or non-string cases
        return False

    # Patterns indicating event listings, theater/cinema schedules, and URLs
    event_patterns = [
        r'https?://[^\s]+',  # URL pattern
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # Date pattern (DD/MM/YYYY or DD-MM-YYYY)
        r'\d{1,2}:\d{1,2}',  # Time pattern (HH:MM)
        r'\b(Theater|Kino|Filmvorführung|Vorstellung|Bühne)\b',  # Theater/Cinema related terms
        r'\b(Ticket|Eintritt|Reservierung|Online-Anmeldung)\b',  # Ticket-related terms
        r'\b(\d{4,5})\b',  # Postal codes (4-5 digits in Austria)
    ]

    # Check if the text matches any of the event-related patterns
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in event_patterns)


# Apply the event listing filter
df = df[~df["text"].apply(is_event_listing)]


# 🟢 Step 6: Convert date column to datetime format (if applicable)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 🟢 Step 7: Save the filtered dataset to a new CSV file
df.to_csv("filtered_articles_expanded.csv", encoding="utf-8", index=False)

print(f"Filtered dataset saved as 'filtered_articles_expanded.csv' with {len(df)} articles.")


