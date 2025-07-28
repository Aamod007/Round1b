import os
import json
import os
import json
import fitz  
import re
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from collections import defaultdict

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SIMILARITY_THRESHOLD = 0.4
TOP_SECTIONS_LIMIT = 5
MIN_PARA_WORD_COUNT = 25
HEADER_FOOTER_MARGIN = 50


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Cleans text by removing excessive whitespace and normalizing."""
    return re.sub(r'\s+', ' ', text).strip()

def intelligent_sectioning(page, doc_name):
    """Segments text using layout info (font size, bold) to identify title-content structures."""
    sections = []
    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
    if not blocks:
        return []

    font_sizes = [span['size'] for block in blocks if 'lines' in block for line in block['lines'] for span in line['spans']]
    if not font_sizes:
        return []
    baseline_font_size = np.percentile(font_sizes, 50)
    
    current_title = "General Information"
    current_content = ""

    for i, block in enumerate(blocks):
        if "lines" not in block or not block.get('lines'):
            continue
        
        if block['bbox'][1] < HEADER_FOOTER_MARGIN or block['bbox'][3] > page.rect.height - HEADER_FOOTER_MARGIN:
            continue

        try:
            first_span = block["lines"][0]["spans"][0]
            font_size = first_span['size']
            is_bold = (first_span['flags'] & 16) > 0
            block_text = clean_text(" ".join(line['spans'][0]['text'] for line in block['lines']))
        except (IndexError, KeyError):
            continue

        is_title = (is_bold and font_size > baseline_font_size + 1) or (font_size > baseline_font_size + 3)

        if is_title and len(block_text.split()) < 20: 
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": clean_text(current_content),
                    "page_number": page.number + 1,
                    "document": doc_name
                })
            current_title = block_text
            current_content = ""
        else:
            current_content += " " + block_text
    
    if current_content:
        sections.append({
            "title": current_title,
            "content": clean_text(current_content),
            "page_number": page.number + 1,
            "document": doc_name
        })
        
    return sections

def process_pdf(pdf_path):
    """Processes a single PDF, extracting sections based on layout."""
    doc_name = os.path.basename(pdf_path)
    all_sections = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            all_sections.extend(intelligent_sectioning(page, doc_name))
        doc.close()
    except Exception as e:
        logging.error(f"Could not process {doc_name} with PyMuPDF: {e}")
    return all_sections

def refine_subsections(section_content, query, model):
    """Extracts the 1-2 most relevant paragraphs from a section."""

    paragraphs = [p.strip() for p in section_content.split('\n') if len(p.strip().split()) >= MIN_PARA_WORD_COUNT]
    if not paragraphs:

        paragraphs = [p.strip() for p in section_content.split('. ') if len(p.strip().split()) >= MIN_PARA_WORD_COUNT]
        if not paragraphs:
            return section_content 

    query_embedding = model.encode(query, convert_to_tensor=True, device='cpu')
    para_embeddings = model.encode(paragraphs, convert_to_tensor=True, device='cpu')
    similarities = util.pytorch_cos_sim(query_embedding, para_embeddings)[0].cpu().numpy()

    top_indices = np.argsort(similarities)[-2:]
    top_paras = [paragraphs[i] for i in top_indices if similarities[i] > SIMILARITY_THRESHOLD]
    
    return " ".join(sorted(top_paras, key=lambda x: section_content.find(x))) # Sort by appearance order

def process_collection(collection_path, model):
    """Main logic to process a single collection and generate all required outputs."""
    input_path = os.path.join(collection_path, 'challenge1b_input.json')
    output_path = os.path.join(collection_path, 'challenge1b_output.json')
    pdfs_dir = os.path.join(collection_path, 'PDFs')

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona_role = input_data['persona']['role']
    job_task = input_data['job_to_be_done']['task']
    query = f"{persona_role}: {job_task}"
    
    all_sections = []
    for doc_info in input_data.get('documents', []):
        pdf_path = os.path.join(pdfs_dir, doc_info['filename'])
        if os.path.exists(pdf_path):
            all_sections.extend(process_pdf(pdf_path))

    if not all_sections:
        logging.warning(f"No sections extracted for {os.path.basename(collection_path)}")
        return

    section_contents = [s['content'] for s in all_sections]
    query_embedding = model.encode(query, convert_to_tensor=True, device='cpu')
    section_embeddings = model.encode(section_contents, convert_to_tensor=True, device='cpu')
    
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)[0]
    scores_above_threshold = cosine_scores > SIMILARITY_THRESHOLD
    
    indices_above_threshold = torch.where(scores_above_threshold)[0]
    scores_filtered = cosine_scores[scores_above_threshold]

    k = min(TOP_SECTIONS_LIMIT, len(scores_filtered))
    
    if k > 0:
        top_results = torch.topk(scores_filtered, k=k)
        top_indices = indices_above_threshold[top_results.indices]
    else:
        top_results = None
        top_indices = []

    extracted_sections = []
    top_sections_for_refinement = []
    if top_results:
        for rank, idx in enumerate(top_indices):
            score = cosine_scores[idx].item()
            meta = all_sections[idx]
            
            extracted_sections.append({
                "document": meta["document"],
                "page_number": meta["page_number"],
                "section_title": meta["title"],
                "importance_rank": rank + 1,
                "similarity_score": round(score, 4)
            })
            top_sections_for_refinement.append(meta)

    refined_subsections = []
    for section in top_sections_for_refinement:
        refined_text = refine_subsections(section['content'], query, model)
        if refined_text:
            refined_subsections.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })

    output_data = {
        "metadata": {
            "input_documents": [d['filename'] for d in input_data.get('documents', [])],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": refined_subsections
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Successfully generated 'challenge1b_output.json' for {os.path.basename(collection_path)}")

def generate_approach_explanation(script_dir, model_name):
    """Generates the single approach_explanation.md file in the root."""
    explanation_path = os.path.join(script_dir, 'approach_explanation.md')
    
    explanation = f"# Approach Explanation\n\n"
    explanation += f"**Model Used:** `{model_name}`\n\n"
    explanation += "## Core Logic & Methodology\n"
    explanation += """
The document intelligence system employs a sophisticated, two-stage pipeline to analyze PDF documents and extract highly relevant, task-focused information.

1.  **Persona and Task Interpretation**: The system begins by creating a semantic query from the `persona` and `job_to_be_done` fields in the input JSON. This query serves as the ground truth for all subsequent relevance scoring.

2.  **Intelligent Sectioning with Layout Analysis**: Each PDF is parsed using **PyMuPDF**, which provides detailed layout information. The script analyzes font attributes (size and weight) to heuristically identify section titles, allowing it to logically segment the document content even without explicit bookmarks. This method is more robust than simple text splitting.

3.  **Multilingual Semantic Ranking (Stage 1)**: The content of each identified section is vectorized using the `paraphrase-multilingual-MiniLM-L12-v2` model. This powerful multilingual model computes the cosine similarity between each section and the persona-driven query. Sections are then ranked by this similarity score, and only those exceeding a **0.4 threshold** are considered for the next stage. The top 5 sections are selected.

4.  **Paragraph-Level Refinement (Stage 2)**: For each of the top-ranked sections, the system performs a second level of analysis. The section's content is broken down into individual paragraphs. Each paragraph is then semantically scored against the same persona query. The 1-2 most relevant paragraphs are extracted and combined to form the final `refined_text`. This ensures the final output is both relevant and concise.

5.  **Multilingual Handling**: The use of a dedicated multilingual sentence transformer ensures that content in all specified languages (English, French, Korean, etc.) is processed and understood natively without the need for a separate translation step, preserving semantic accuracy.

6.  **Final Output Generation**: The top 5 sections and their corresponding refined text snippets are compiled into the `generated_output.json`. A single, consolidated `approach_explanation.md` is also created in the root directory to document this methodology.
"""
    with open(explanation_path, 'w', encoding='utf-8') as f:
        f.write(explanation)
    logging.info(f"Successfully generated 'approach_explanation.md'")

def main():
    """Finds and processes all collections based on the defined logic."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logging.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    logging.info("Model loaded successfully.")

    for item in os.listdir(script_dir):
        collection_path = os.path.join(script_dir, item)
        if os.path.isdir(collection_path) and item.startswith('Collection'):
            logging.info(f"--- Processing {item} ---")
            try:
                process_collection(collection_path, model)
            except Exception as e:
                logging.error(f"Failed to process {item}. Error: {e}", exc_info=True)
    
    generate_approach_explanation(script_dir, MODEL_NAME)

if __name__ == '__main__':
    main()
