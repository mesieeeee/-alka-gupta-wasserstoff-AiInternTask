import os
import time
import psutil
import PyPDF2
import pymongo
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# MongoDB connection setup
def connect_to_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
    db = client["pdf_database"]  # Database name
    collection = db["pdf_metadata"]  # Collection name
    return collection

# Preprocess text (as before)
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# TextRank summarization (as before)
def textrank_summary(text, max_percent=0.05):
    text = re.sub(r'\n+', ' ', text)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    num_sentences = max(1, int(len(sentences) * max_percent))
    ranked_indices = np.argsort(sentence_scores)[-num_sentences:]
    summary = '. '.join([sentences[i].strip() for i in sorted(ranked_indices)])
    return summary

# Keyword extraction (as before)
def extract_keywords(text, n_keywords=10, stop_words=None):
    if stop_words is None:
        stop_words = [
            'the', 'and', 'is', 'in', 'to', 'a', 'that', 'of', 'it', 'on', 'for',
            'with', 'as', 'was', 'at', 'by', 'an', 'this', 'are', 'be', 'or', 'if'
        ]
    cleaned_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    keywords_scores = dict(zip(feature_names, scores))
    sorted_keywords = sorted(keywords_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in sorted_keywords[:n_keywords]]
    return top_keywords

# Categorize PDF
def categorize_pdf(file_path):
    try:
        file_size = os.path.getsize(file_path) / 1024  # KB
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
        summary = textrank_summary(text, max_percent=0.09)
        keywords = extract_keywords(text, n_keywords=10)
        category = 'short' if num_pages <= 10 else 'medium' if num_pages <= 30 else 'long'
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'category': category,
            'num_pages': num_pages,
            'size_kb': file_size,
            'summary': summary,
            'summary_length': len(summary),
            'keywords': keywords,
            'keywords_length': len(keywords),
            'text_length': len(text)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Store PDF metadata in MongoDB
def store_pdf_metadata(metadata, collection):
    try:
        inserted_id = collection.insert_one(metadata).inserted_id
        print(f"Document inserted with ID: {inserted_id}")
    except Exception as e:
        print(f"Error storing metadata in MongoDB: {e}")

# Track system resource usage
def log_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_info.percent}%")

# Ingest PDFs and track performance
def ingest_pdfs(folder_path):
    pdf_files = {f: os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')}
    collection = connect_to_mongo()

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        for file_name, file_path in pdf_files.items():
            futures.append(executor.submit(categorize_pdf, file_path))

        for future in futures:
            result = future.result()
            if result:
                store_pdf_metadata(result, collection)
            
            # Log resource usage for each processed file
            log_system_metrics()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print total execution time and number of PDFs processed
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total PDFs processed: {len(pdf_files)}")

# Replace with your folder path containing PDFs
folder_path = 'C:/Users/madam/OneDrive/Desktop/pdf_task'
# Ingest PDFs from the folder and store in MongoDB
ingest_pdfs(folder_path)
