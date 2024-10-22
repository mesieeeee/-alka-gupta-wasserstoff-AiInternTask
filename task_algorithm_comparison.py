import os

# Example path to your PDFs folder in Google Drive
folder_path = '/content/drive/MyDrive/pdf_task'

# Listing files to confirm the folder contents
pdf_files = os.listdir(folder_path)
print(pdf_files)

import os
import PyPDF2
from concurrent.futures import ThreadPoolExecutor

# Function to categorize PDF based on length
def categorize_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        if num_pages <= 10:
            return 'short', num_pages
        elif num_pages <= 30:
            return 'medium', num_pages
        else:
            return 'long', num_pages

# Ingest all PDFs from the folder
def ingest_pdfs(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

    with ThreadPoolExecutor() as executor:
        results = executor.map(categorize_pdf, pdf_files)

    return list(results)

ingested_pdfs = ingest_pdfs(folder_path)
print(ingested_pdfs)

print(ingested_pdfs)

print(pdf_files)


os.listdir()

ingested_pdfs.index(('medium', 11))

import pymongo
import os
from pymongo import MongoClient


# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_database']
collection = db['pdf_metadata']

# Function to store metadata
def store_metadata(file_path, category, num_pages):
    metadata = {
        'document_name': os.path.basename(file_path),
        'path': file_path,
        'size': os.path.getsize(file_path),
        'category': category,
        'num_pages': num_pages,
        'summary': None,
        'keywords': None
    }
    collection.insert_one(metadata)

# Example of inserting metadata
for pdf in ingested_pdfs:
    category, num_pages = pdf[0], pdf[1]
    file_path = pdf_files[ingested_pdfs.index(pdf)]
    store_metadata(file_path, category, num_pages)

# Function to update document after summarization
def update_document_summary(file_path, summary, keywords):
    collection.update_one(
        {'path': file_path},
        {'$set': {'summary': summary, 'keywords': keywords}}
    )

import os
import PyPDF2
from concurrent.futures import ThreadPoolExecutor

# Function to categorize PDF based on length and size
def categorize_pdf(file_path):
    try:
        # Get file size in KB
        file_size = os.path.getsize(file_path) / 1024  # KB

        # Open and read PDF
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

        # Categorize PDF based on page count
        if num_pages <= 10:
            category = 'short'
        elif num_pages <= 30:
            category = 'medium'
        else:
            category = 'long'

        # Return result as dictionary
        return {
            'file_path': file_path,
            'category': category,
            'num_pages': num_pages,
            'size_kb': file_size
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Ingest all PDFs from the folder and store as dictionary
def ingest_pdfs(folder_path):
    pdf_files = {f: os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')}

    # Dictionary to store the final result with file name as key
    results_dict = {}

    # Process PDFs concurrently
    with ThreadPoolExecutor() as executor:
        for file_name, result in zip(pdf_files.keys(), executor.map(categorize_pdf, pdf_files.values())):
            if result:
                results_dict[file_name] = result

    return results_dict

# Example usage (adjust the folder path accordingly)
folder_path = '/content/drive/MyDrive/pdf_task'

# Ingest PDFs from the folder
ingested_pdfs = ingest_pdfs(folder_path)
print(ingested_pdfs)

from bson import ObjectId  # for MongoDB IDs

# MongoDB connection
def connect_to_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")  # Adjust with your MongoDB connection URI
    db = client["pdf_database"]
    collection = db["pdf_metadata"]
    return collection

# Store PDF metadata in MongoDB
def store_pdf_metadata(metadata, collection):
    try:
        # Insert metadata into MongoDB
        inserted_id = collection.insert_one(metadata).inserted_id
        print(f"Document inserted with ID: {inserted_id}")
    except Exception as e:
        print(f"Error storing metadata in MongoDB: {e}")

# Ingest all PDFs from the folder, categorize them, and store metadata in MongoDB
def ingest_pdfs(folder_path):
    pdf_files = {f: os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')}

    # Dictionary to store the final result with file name as key
    results_dict = {}

    # Connect to MongoDB
    collection = connect_to_mongo()

    # Process PDFs concurrently
    with ThreadPoolExecutor() as executor:
        for file_name, result in zip(pdf_files.keys(), executor.map(categorize_pdf, pdf_files.values())):
            if result:
                results_dict[file_name] = result
                # Store the PDF metadata in MongoDB
                store_pdf_metadata(result, collection)

    return results_dict

import os
import PyPDF2
import pymongo
from concurrent.futures import ThreadPoolExecutor
from bson import ObjectId

# MongoDB connection setup
def connect_to_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
    db = client["pdf_database"]  # Database name
    collection = db["pdf_metadata"]  # Collection name
    return collection

# Function to categorize PDF based on length and size
def categorize_pdf(file_path):
    try:
        # Get file size in KB
        file_size = os.path.getsize(file_path) / 1024  # KB

        # Open and read PDF
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

        # Categorize PDF based on page count
        if num_pages <= 10:
            category = 'short'
        elif num_pages <= 30:
            category = 'medium'
        else:
            category = 'long'

        # Return result as dictionary
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'size_kb': file_size,
            'num_pages': num_pages,
            'category': category
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Store PDF metadata in MongoDB
def store_pdf_metadata(metadata, collection):
    try:
        # Insert metadata into MongoDB
        inserted_id = collection.insert_one(metadata).inserted_id
        return inserted_id
    except Exception as e:
        print(f"Error storing metadata in MongoDB: {e}")
        return None

# Update MongoDB record with summary and keywords
def update_pdf_metadata(collection, doc_id, summary, keywords):
    try:
        # Update the MongoDB document with summary and keywords
        collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"summary": summary, "keywords": keywords}}
        )
        print(f"Document {doc_id} updated with summary and keywords.")
    except Exception as e:
        print(f"Error updating document {doc_id}: {e}")

# Ingest all PDFs from the folder and store as dictionary, then store metadata in MongoDB
def ingest_pdfs(folder_path):
    pdf_files = {f: os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')}

    # Dictionary to store the final result with file name as key
    results_dict = {}

    # Connect to MongoDB
    collection = connect_to_mongo()

    # Process PDFs concurrently
    with ThreadPoolExecutor() as executor:
        for file_name, result in zip(pdf_files.keys(), executor.map(categorize_pdf, pdf_files.values())):
            if result:
                # Store metadata in MongoDB
                doc_id = store_pdf_metadata(result, collection)
                if doc_id:
                    results_dict[file_name] = {"metadata": result, "doc_id": doc_id}

    return results_dict

# Example function to simulate summarization and keyword extraction (replace with actual processing logic)
def generate_summary_and_keywords(file_path):
    # Placeholder for actual summarization and keyword extraction logic
    summary = f"Generated summary for {file_path}"
    keywords = ["keyword1", "keyword2", "keyword3"]
    return summary, keywords

# Update metadata in MongoDB after post-processing (summarization and keyword extraction)
def post_process_pdfs(results_dict):
    collection = connect_to_mongo()

    # Iterate over the stored results
    for file_name, data in results_dict.items():
        file_path = data['metadata']['file_path']
        doc_id = data['doc_id']

        # Generate summary and keywords
        summary, keywords = generate_summary_and_keywords(file_path)

        # Update MongoDB with the generated summary and keywords
        update_pdf_metadata(collection, doc_id, summary, keywords)

# Example usage (adjust the folder path accordingly)
folder_path = '/content/drive/MyDrive/pdf_task'

# Step 1: Ingest PDFs from the folder and store metadata in MongoDB
ingested_pdfs = ingest_pdfs(folder_path)

# Step 2: After processing (summarization & keyword extraction), update MongoDB with the results
post_process_pdfs(ingested_pdfs)

import pdfplumber

def extract_text_from_pdf(file_path):
    """Extracts text from each page of the given PDF file."""
    text = ''

    # Open the PDF file
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            page_text = page.extract_text()

            # Handle case where text extraction fails
            if page_text:
                text += f'--- Page {page_number + 1} ---\n'
                text += page_text + '\n'
            else:
                text += f'--- Page {page_number + 1} ---\n'
                text += 'No text found on this page.\n'

    return text

# Example usage
file_path = '/content/admin_judgement_file_judgement_pdf_1960_volume 1_Part I_great indian motor works ltd., and another_their employees and others_1699336355.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(file_path)
print(extracted_text)

# Example usage to save to a file
output_file_path = 'extracted_text1.txt'  # Output file path
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(extracted_text)

print(f'Text extracted and saved to {output_file_path}.')

import pdfplumber

def extract_text_from_pdf(file_path):
    """Extracts text from each page of the given PDF file."""
    text = ''

    # Open the PDF file
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            page_text = page.extract_text()

            # Handle case where text extraction fails
            if page_text:
                text += f'--- Page {page_number + 1} ---\n'
                text += page_text + '\n'
            else:
                text += f'--- Page {page_number + 1} ---\n'
                text += 'No text found on this page.\n'

    return text

# Example usage
file_path = '/content/Circular Orders (Supplement).pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(file_path)
print(extracted_text)

# Example usage to save to a file
output_file_path = 'extracted_text2.txt'  # Output file path
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(extracted_text)

print(f'Text extracted and saved to {output_file_path}.')

import pdfplumber

def extract_text_from_pdf(file_path):
    """Extracts text from each page of the given PDF file."""
    text = ''

    # Open the PDF file
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            page_text = page.extract_text()

            # Handle case where text extraction fails
            if page_text:
                text += f'--- Page {page_number + 1} ---\n'
                text += page_text + '\n'
            else:
                text += f'--- Page {page_number + 1} ---\n'
                text += 'No text found on this page.\n'

    return text

# Example usage
file_path = '/content/admin_judgement_file_judgement_pdf_1952_volume 1_Part I_the state of bihar_maharajadhiraja sir kameshwar singh of darbhanga and others_1698318448.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(file_path)
print(extracted_text)

# Example usage to save to a file
output_file_path = 'extracted_text3.txt'  # Output file path
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(extracted_text)

print(f'Text extracted and saved to {output_file_path}.')

text = '''purpose, the non obstante clause in article 31 ( 4) over-
rides all other provisions in the Constitution including
the List of the Seventh Schedule, whereas a law which
falls within the purview of article 31-A could only
prevail over "the foregoing provisions of this Part".
Now, the three impugned statues fall within the
" ambit of both article 31 ( 4) and articles 31-A and 31-B.
Putting aside the later articles for the moment, it is
plain that, under article 31 ( 4), the three impugned
statutes are protected from attack in any court on the
• ground that they contravene the provisions of article
31(2). These provisions, so far as they are material
here, ~ ·e (i) that a law with respect to acquisition of
• property ~hould authorize acquisition only for a
...., public purpose and (ii) that such law should provide
for compensation, etc. Mr. Das, while admitting tha'''

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses the text by removing punctuation, newlines, and converting to lowercase."""
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def textrank_summary(text, max_percent=0.05):
    # Remove newlines from the text to ensure cleaner sentences
    text = re.sub(r'\n+', ' ', text)  # Removing newline characters before processing

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Preprocess sentences by removing unwanted characters and lowercase
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences using a simple graph centrality measure (sum of similarities)
    sentence_scores = np.sum(similarity_matrix, axis=1)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])  # Strip removes any leading/trailing spaces

    return summary

# Read text from the file
file_path = 'extracted_text1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = textrank_summary(text, max_percent=0.09)  # Adjust max_percent as needed
print("Generated Summary:")
print(summary)
print(len(summary))

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses the text by removing punctuation, newlines, and converting to lowercase."""
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def textrank_summary(text, max_percent=0.05):
    # Remove newlines from the text to ensure cleaner sentences
    text = re.sub(r'\n+', ' ', text)  # Removing newline characters before processing

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Preprocess sentences by removing unwanted characters and lowercase
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences using a simple graph centrality measure (sum of similarities)
    sentence_scores = np.sum(similarity_matrix, axis=1)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])  # Strip removes any leading/trailing spaces

    return summary

# Read text from the file
file_path = 'extracted_text2.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = textrank_summary(text, max_percent=0.09)  # Adjust max_percent as needed
print("Generated Summary:")
print(summary)
print(len(summary))

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses the text by removing punctuation, newlines, and converting to lowercase."""
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def textrank_summary(text, max_percent=0.05):
    # Remove newlines from the text to ensure cleaner sentences
    text = re.sub(r'\n+', ' ', text)  # Removing newline characters before processing

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Preprocess sentences by removing unwanted characters and lowercase
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences using a simple graph centrality measure (sum of similarities)
    sentence_scores = np.sum(similarity_matrix, axis=1)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])  # Strip removes any leading/trailing spaces

    return summary

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = textrank_summary(text, max_percent=0.09)  # Adjust max_percent as needed
print("Generated Summary:")
print(summary)
print(len(summary))

from rouge_score import rouge_scorer

def calculate_rouge(reference_summary, generated_summary):
    """Calculate ROUGE scores between a reference summary and a generated summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# Example summaries
reference_summary = summary

generated_summary = """Putting aside the later articles for the moment, it is plain that, under article 31 ( 4), the three impugned statutes are protected from attack in any court on the • ground that they contravene the provisions of article 31(2)"""

# Calculate ROUGE scores
rouge_scores = calculate_rouge(reference_summary, generated_summary)

# Print the results
for metric, score in rouge_scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-Score={score.fmeasure:.4f}")

"""for comparison, try to implement the other approaches and let's compare the 2 sumamries using rouge score??

Cosine Similarity with TF-IDF:
How to Use: Compare the cosine similarity between your summary and the entire document. If the summary represents the core concepts of the document well, it should have a high cosine similarity.
Why Use It: This checks whether the summary captures the essence of the document without being redundant.
3. Comparative Evaluation
Compare your current extractive approach (TextRank) with other algorithms and models (without relying on pre-built pipelines), to check for improvements in quality:

LexRank: An alternative graph-based algorithm that uses a similar principle but applies eigenvector centrality.
LDA-based Summarization: Using Latent Dirichlet Allocation (LDA) for topic modeling, followed by extracting representative sentences for each topic.
Custom Scoring Mechanism: You could implement a custom scoring system where sentence importance is determined by word frequency, sentence position, or keywords.
4. Document Length Handling
Scalability for Long Documents: For longer documents (like legal texts), check how well your summarizer handles them. If the summaries for longer texts are too short, or they miss key concepts, you might need to tune parameters like max_percent in your script to vary the length dynamically.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def summarize_with_tfidf(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Sentence splitting
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Score sentences based on their TF-IDF scores
    scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Calculate number of sentences to extract based on document length
    num_sentences = max(1, int(len(sentences) * max_percent))

    ranked_indices = np.argsort(scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = summarize_with_tfidf(text)
print("Generated Summary:")
print(summary)
print(len(summary))

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def summarize_with_tfidf(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Sentence splitting
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Score sentences based on their TF-IDF scores
    scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Calculate number of sentences to extract based on document length
    num_sentences = max(1, int(len(sentences) * max_percent))

    ranked_indices = np.argsort(scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text2.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = summarize_with_tfidf(text)
print("Generated Summary:")
print(summary)
print(len(summary))

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def summarize_with_tfidf(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Sentence splitting
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Score sentences based on their TF-IDF scores
    scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Calculate number of sentences to extract based on document length
    num_sentences = max(1, int(len(sentences) * max_percent))

    ranked_indices = np.argsort(scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = summarize_with_tfidf(text)
print("Generated Summary:")
print(summary)
print(len(summary))

from collections import Counter

def frequency_based_summary(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Count word frequencies
    words = text.split()
    word_frequencies = Counter(words)

    # Score each sentence by the sum of word frequencies
    sentence_scores = []
    for sentence in sentences:
        score = sum([word_frequencies[word] for word in sentence.split()])
        sentence_scores.append(score)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences based on their score
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = frequency_based_summary(text)
print("Generated Summary:")
print(summary)
print(len(summary))

from collections import Counter

def frequency_based_summary(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Count word frequencies
    words = text.split()
    word_frequencies = Counter(words)

    # Score each sentence by the sum of word frequencies
    sentence_scores = []
    for sentence in sentences:
        score = sum([word_frequencies[word] for word in sentence.split()])
        sentence_scores.append(score)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences based on their score
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text2.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = frequency_based_summary(text)
print("Generated Summary:")
print(summary)
print(len(summary))

from collections import Counter

def frequency_based_summary(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Count word frequencies
    words = text.split()
    word_frequencies = Counter(words)

    # Score each sentence by the sum of word frequencies
    sentence_scores = []
    for sentence in sentences:
        score = sum([word_frequencies[word] for word in sentence.split()])
        sentence_scores.append(score)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences based on their score
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = frequency_based_summary(text)
print("Generated Summary:")
print(summary)
print(len(summary))

"""compare rouge text rank, tfidf vectorizer, frquency based summarization"""

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses the text by removing punctuation, newlines, and converting to lowercase."""
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def textrank_summary(text, max_percent=0.05):
    # Remove newlines from the text to ensure cleaner sentences
    text = re.sub(r'\n+', ' ', text)  # Removing newline characters before processing

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Preprocess sentences by removing unwanted characters and lowercase
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences using a simple graph centrality measure (sum of similarities)
    sentence_scores = np.sum(similarity_matrix, axis=1)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])  # Strip removes any leading/trailing spaces

    return summary

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary = textrank_summary(text, max_percent=0.09)  # Adjust max_percent as needed
print("Generated Summary:")
print(summary)
print(len(summary))

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def summarize_with_tfidf(text, max_percent=0.05):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Sentence splitting
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Score sentences based on their TF-IDF scores
    scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Calculate number of sentences to extract based on document length
    num_sentences = max(1, int(len(sentences) * max_percent))

    ranked_indices = np.argsort(scores)[-num_sentences:]
    summary = '. '.join([sentences[i] for i in sorted(ranked_indices)])
    return summary

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(len(text))

# Generate summary
summary1 = summarize_with_tfidf(text)
print("Generated Summary:")
print(summary1)
print(len(summary1))

from rouge_score import rouge_scorer

def calculate_rouge(reference_summary, generated_summary):
    """Calculate ROUGE scores between a reference summary and a generated summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# Example summaries
reference_summary = summary

generated_summary = summary1

# Calculate ROUGE scores
rouge_scores = calculate_rouge(reference_summary, generated_summary)

# Print the results
for metric, score in rouge_scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-Score={score.fmeasure:.4f}")

from rouge_score import rouge_scorer

def calculate_rouge(reference_summary, generated_summary):
    """Calculate ROUGE scores between a reference summary and a generated summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# Example summaries
reference_summary = summary

generated_summary = summary1

# Calculate ROUGE scores
rouge_scores = calculate_rouge(reference_summary, generated_summary)

# Print the results
for metric, score in rouge_scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-Score={score.fmeasure:.4f}")

from rouge_score import rouge_scorer

def calculate_rouge(reference_summary, generated_summary):
    """Calculate ROUGE scores between a reference summary and a generated summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# Example summaries
reference_summary = summary

generated_summary = summary1

# Calculate ROUGE scores
rouge_scores = calculate_rouge(reference_summary, generated_summary)

# Print the results
for metric, score in rouge_scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-Score={score.fmeasure:.4f}")

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses the text by removing punctuation, newlines, and converting to lowercase."""
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

def textrank_summary(text, max_percent=0.05):
    """Generates a summary using the TextRank algorithm."""
    # Remove newlines from the text to ensure cleaner sentences
    text = re.sub(r'\n+', ' ', text)

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Preprocess sentences
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences using a simple graph centrality measure (sum of similarities)
    sentence_scores = np.sum(similarity_matrix, axis=1)

    # Calculate number of sentences to extract
    num_sentences = max(1, int(len(sentences) * max_percent))

    # Extract top sentences
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])  # Strip removes any leading/trailing spaces

    return summary

def extract_keywords(text, n_keywords=10, stop_words=None):
    """Extracts non-generic, domain-specific keywords from the text."""
    if stop_words is None:
        stop_words = [
            'the', 'and', 'is', 'in', 'to', 'a', 'that', 'of', 'it', 'on', 'for',
            'with', 'as', 'was', 'at', 'by', 'an', 'this', 'are', 'be', 'or', 'if'
        ]  # Add more domain-specific stop words if needed

    # Preprocess text for keyword extraction
    cleaned_text = preprocess_text(text)

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])

    # Get feature names and their corresponding scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()

    # Create a dictionary of keywords and their scores
    keywords_scores = dict(zip(feature_names, scores))

    # Sort keywords by their scores and select the top N keywords
    sorted_keywords = sorted(keywords_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in sorted_keywords[:n_keywords]]

    return top_keywords

# Read text from the file
file_path = 'extracted_text3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    print(f"Text Length: {len(text)} characters")

# Generate summary
summary = textrank_summary(text, max_percent=0.09)  # Adjust max_percent as needed
print("Generated Summary:")
print(summary)
print(f"Summary Length: {len(summary)} characters")

# Extract keywords
keywords = extract_keywords(text, n_keywords=10)  # Adjust n_keywords as needed
print("Extracted Keywords:")
print(keywords)

