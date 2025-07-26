# Approach Explanation: Persona-Driven Document Intelligence

## Introduction

This document outlines the methodology used to build an intelligent document analysis system for the Adobe India Hackathon 2025. The system is designed to extract and prioritize relevant sections from a collection of PDF documents based on a given persona and their job-to-be-done. The goal is to deliver a ranked list of the most relevant content in a structured JSON format.

## Methodology

Our approach is a multi-stage pipeline that processes, filters, and ranks document sections to meet the user's specific needs.

### 1. Document Processing and Text Extraction

The system begins by processing each PDF document in the input collection using the **PyMuPDF (fitz)** library, which is highly efficient for text extraction. For each page in a PDF, we extract the raw text content, which serves as the basis for all subsequent analysis.

### 2. Section Segmentation and Validation

Once the text is extracted, we segment it into meaningful sections. For the given recipe book test case, a "section" is a single recipe. Our segmentation logic uses regular expressions to split the text into blocks based on common recipe separators (multiple newlines).

Each block is then validated to ensure it is a valid recipe. This is done by checking for the presence of keywords like "Ingredients," "Instructions," "Method," or "Preparation." We also validate the title of the recipe to ensure it is descriptive and not a stray bullet point or non-textual element. This ensures that only well-formed recipes are considered for ranking. To prevent redundancy, we maintain a set of seen sections and discard any duplicates.

### 3. Multi-Layered Filtering and Ranking

This is the core of our intelligent analysis. We use a combination of keyword filtering and semantic ranking to identify the most relevant sections.

*   **Vegetarian Filtering:** We first apply a strict filter to remove any non-vegetarian recipes. This is achieved by using a comprehensive list of non-vegetarian keywords (e.g., "chicken," "beef," "pancetta").

*   **Semantic Ranking:** The remaining vegetarian sections are then ranked based on their relevance to the user's query (persona + job-to-be-done). We use the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model to generate embeddings for the query and each section. The cosine similarity between the embeddings determines the initial relevance score.

*   **Contextual Boosting and Penalizing:** To further refine the ranking, we apply a set of contextual rules:
    *   **Gluten-Free Boost:** Recipes containing the phrase "gluten-free" are given a score boost, directly addressing a key requirement of the job-to-be-done.
    *   **Non-Dinner Penalty:** Recipes for items that are not suitable for a dinner buffet (e.g., "sandwich," "breakfast," "smoothie") are penalized by reducing their relevance score.

### 4. Language Detection and Output Generation

For each valid section, we use the `langdetect` library to identify the language of the content. This metadata is included in the final output for better traceability and multilingual support.

The final ranked and filtered list of sections is then formatted into the specified JSON structure, including metadata, extracted sections, and subsection analysis. The entire process is logged for better debugging and monitoring.
