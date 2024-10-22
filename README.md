# PDF Metadata Ingestion and Analysis

This project automates the ingestion of PDF files from a specified folder, extracts metadata, summarizes the content, extracts keywords, and stores the results in a MongoDB database. It utilizes concurrent processing for efficiency and tracks system resource usage during execution.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Code Explanation](#code-explanation)

## Features
- Automatically processes PDF files in a designated folder.
- Extracts text, generates summaries using the TextRank algorithm, and identifies keywords.
- Categorizes PDFs based on page count and size.
- Stores extracted metadata in a MongoDB database.
- Monitors system performance metrics (CPU and memory usage) during execution.
- Supports concurrent processing for improved speed.

## System Requirements
- Python 3.6 or higher
- MongoDB (installed locally or accessible via a connection string)
- Required Python libraries:
  - `PyPDF2`
  - `pymongo`
  - `numpy`
  - `scikit-learn`
  - `psutil`
  - `concurrent.futures`
  
You can install the required libraries using pip:
```bash
pip install PyPDF2 pymongo numpy scikit-learn psutil

Setup Instructions
git clone https://your-repository-url.git
cd pdf-metadata-ingestion
Install dependencies: Make sure you have all the necessary libraries as mentioned in the system requirements.

Setup MongoDB: Ensure that MongoDB is installed and running on your local machine. The default connection string used in the code is mongodb://localhost:27017/. If your MongoDB server is running on a different host or port, adjust the connection string accordingly in the connect_to_mongo function.

Place PDF Files: Put all your PDF files in a designated folder (e.g., C:/Users/madam/OneDrive/Desktop/pdf_task).

Configure Folder Path: Modify the folder_path variable in the script to point to the folder containing your PDFs.

Code Explanation
MongoDB Connection: The connect_to_mongo function establishes a connection to a MongoDB database and returns the specified collection.

Text Preprocessing: The preprocess_text function cleans the text by removing punctuation and converting it to lowercase.

TextRank Summary: The textrank_summary function generates a summary of the input text using the TextRank algorithm, leveraging the TfidfVectorizer to compute sentence similarities.

Keyword Extraction: The extract_keywords function extracts relevant keywords from the text, utilizing TF-IDF scores to identify the most significant terms.

PDF Categorization: The categorize_pdf function opens a PDF file, reads its content, extracts text, generates a summary and keywords, and categorizes the PDF based on its page count.

Metadata Storage: The store_pdf_metadata function inserts the processed metadata into the MongoDB collection, the terms that were included in meta data are:-
      _id
      file_name
      file_path
      category
      num_pages
      size_kb
      summary
      summary_length
      keywords
      keywords_length
      text_length

System Resource Logging: The log_system_metrics function logs the CPU and memory usage during the execution.

PDF Ingestion: The ingest_pdfs function coordinates the entire process. It collects PDF files, processes them concurrently using parallel processing through threads, utilizing system resourecs, maximizing speed and efficency, stores metadata in MongoDB, and logs system metrics.

Why TextRank was Chosen
Based on the above comparisons, TextRank was chosen as the main summarization algorithm because:

Adaptability: It works well with documents of all lengths (short, medium, and long), providing summaries that match the size of the document, which is crucial for the diverse nature of PDFs we are working with.
Consistency: The summaries produced by TextRank are more coherent and contextually accurate compared to other algorithms like n-grams or clustering.
Metrics: When evaluated with ROUGE metrics, TextRank performed well in both precision and recall, particularly for longer documents, where other methods struggled.
Efficiency: TextRank is an unsupervised method that doesn’t require training data, making it a computationally efficient choice for this task

