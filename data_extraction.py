import xml.etree.ElementTree as ET
import pandas as pd
import re

def extract_from_xml(file_path):
    """Extract article title and text from a Wikipedia XML dump file."""
    articles = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Namespace may vary, so let's find it dynamically
    namespace = ''
    for elem in root.iter():
        if '}' in elem.tag:
            namespace = elem.tag[:elem.tag.index('}')+1]
            break

    for page in root.findall(f'.//{namespace}page'):
        title = page.find(f'{namespace}title').text
        text = page.find(f'.//{namespace}text').text
        if text:  # Ensure text is not None
            articles.append({'title': title, 'text': text})
    return articles

def combine_articles(file_paths):
    """Combine articles from multiple XML files into a single DataFrame."""
    all_articles = []
    for file_path in file_paths:
        articles = extract_from_xml(file_path)
        all_articles.extend(articles)
    return pd.DataFrame(all_articles)

# List of your XML file paths
file_paths = [
    'enwiki-20250220-pages-articles1.xml-p1p41242'
]

# Combine articles from all files
df = combine_articles(file_paths)
df.to_csv("cleaned_data.csv",index=False)
