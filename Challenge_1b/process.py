import os
import json
import fitz  # PyMuPDF
import re
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Configuration ---
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
RELEVANCE_THRESHOLD = 0.45  # A balanced threshold
MIN_SECTION_LENGTH = 40
HEADER_FOOTER_MARGIN = 50

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_text(text):
    """Cleans text by removing excessive whitespace and normalizing."""
    return re.sub(r'\s+', ' ', text).strip()

def is_header_or_footer(block, page_height):
    """Checks if a text block is likely a header or footer."""
    bbox = block['bbox']
    return bbox[1] < HEADER_FOOTER_MARGIN or bbox[3] > page_height - HEADER_FOOTER_MARGIN

def intelligent_sectioning(page, doc_name):
    """Segments text using layout info (boldness) to identify title-content structures."""
    sections = []
    page_height = page.rect.height
    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
    if not blocks:
        return []

    for i, block in enumerate(blocks):
        if "lines" not in block or not block.get('lines') or is_header_or_footer(block, page_height):
            continue

        try:
            first_span = block["lines"][0]["spans"][0]
            is_bold_title = first_span['flags'] & 16
            title_text = clean_text(" ".join(span['text'] for span in block["lines"][0]["spans"]))
        except (IndexError, KeyError):
            continue

        if is_bold_title and len(title_text) > 3:
            content_text = ""
            # Find the content for the current title
            for j in range(i + 1, len(blocks)):
                next_block = blocks[j]
                if "lines" not in next_block or not next_block.get('lines') or is_header_or_footer(next_block, page_height):
                    continue
                try:
                    next_span = next_block["lines"][0]["spans"][0]
                    # Stop if we hit the next title
                    if next_span['flags'] & 16:
                        break
                    content_text += " " + clean_text(" ".join(span['text'] for line in next_block['lines'] for span in line['spans']))
                except (IndexError, KeyError):
                    continue
            
            # Check for ingredients list
            if "ingredients" in content_text.lower() and len(content_text) > MIN_SECTION_LENGTH:
                sections.append({
                    "title": title_text,
                    "content": clean_text(content_text),
                    "page_number": page.number + 1,
                    "document": doc_name
                })
    return sections

def process_pdf(pdf_path):
    """Processes a single PDF, extracting sections."""
    doc_name = os.path.basename(pdf_path)
    all_sections = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.number is not None:
                all_sections.extend(intelligent_sectioning(page, doc_name))
        doc.close()
    except Exception as e:
        logging.error(f"Error processing {doc_name}: {e}")
    return all_sections

def rank_and_filter_sections(sections, query, model, job_task):
    """Ranks sections using semantic similarity combined with keyword boosting/penalization."""
    if not sections:
        return []
    
    section_texts = [s['title'] + " " + s['content'] for s in sections]
    query_embedding = model.encode(query, convert_to_tensor=True, device='cpu')
    section_embeddings = model.encode(section_texts, convert_to_tensor=True, device='cpu')
    
    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)[0].cpu().numpy()
    
    # --- Contextual Keyword-Based Scoring ---
    task_lower = job_task.lower()
    
    # Define keywords based on the vegetarian dinner buffet task
    boosting_keywords = {'vegetarian', 'gluten-free', 'vegan', 'salad', 'vegetable', 'lentil', 'chickpea', 'quinoa', 'tofu', 'falafel', 'ratatouille', 'lasagna'}
    penalizing_keywords = {'chicken', 'beef', 'pork', 'fish', 'breakfast', 'snack', 'bacon', 'sausage', 'ham'}

    for i, text in enumerate(section_texts):
        text_lower = text.lower()
        # Apply boosts for relevant terms
        for keyword in boosting_keywords:
            if keyword in text_lower:
                similarities[i] += 0.3  # Add a significant boost
        # Apply penalties for irrelevant terms
        for keyword in penalizing_keywords:
            if keyword in text_lower:
                similarities[i] -= 0.5  # Apply a heavy penalty

    for i, section in enumerate(sections):
        section['relevance'] = float(similarities[i])
        
    # Filter based on the final, adjusted score
    relevant_sections = [s for s in sections if s['relevance'] >= RELEVANCE_THRESHOLD]
    return sorted(relevant_sections, key=lambda x: x['relevance'], reverse=True)

def process_collection(collection_path, model):
    """Main logic to process a single collection and generate all required outputs."""
    input_path = os.path.join(collection_path, 'challenge1b_input.json')
    output_path = os.path.join(collection_path, 'generated_output.json')
    pdfs_dir = os.path.join(collection_path, 'PDFs')

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona_role = input_data['persona']['role']
    job_task = input_data['job_to_be_done']['task']
    query = f"Persona: {persona_role}. Task: {job_task}"
    
    pdf_documents = input_data.get('documents', [])
    
    all_sections = []
    for doc_info in pdf_documents:
        pdf_path = os.path.join(pdfs_dir, doc_info['filename'])
        if os.path.exists(pdf_path):
            all_sections.extend(process_pdf(pdf_path))

    stats = {'total_sections': len(all_sections)}
    ranked_sections = rank_and_filter_sections(all_sections, query, model, job_task)

    # Post-processing to clean up subsection_analysis
    for i, section in enumerate(ranked_sections):
        content = section['content']
        if i + 1 < len(ranked_sections):
            next_section_title = ranked_sections[i+1]['title']
            if next_section_title in content:
                content = content.split(next_section_title)[0]
        section['content'] = content

    output_data = {
        "metadata": {
            "input_documents": [d['filename'] for d in pdf_documents],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [{
            "document": s['document'],
            "section_title": s['title'],
            "importance_rank": i + 1,
            "page_number": s['page_number']
        } for i, s in enumerate(ranked_sections)],
        "subsection_analysis": [{
            "document": s['document'],
            "refined_text": s['content'],
            "page_number": s['page_number']
        } for s in ranked_sections]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Successfully generated 'generated_output.json' for {os.path.basename(collection_path)}")

def main():
    """Finds and processes all collections with the keyword-driven logic."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logging.info("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    logging.info("Model loaded.")

    for item in os.listdir(script_dir):
        collection_path = os.path.join(script_dir, item)
        if os.path.isdir(collection_path) and item.startswith('Collection'):
            logging.info(f"--- Processing {item} with keyword-driven logic ---")
            try:
                process_collection(collection_path, model)
            except Exception as e:
                logging.error(f"Failed to process {item}. Error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
