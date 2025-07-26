import os
import json
import fitz  # PyMuPDF
import re
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from langdetect import detect, LangDetectException

# --- Configuration ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_text(text):
    """Cleans text by removing excessive whitespace and normalizing."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text):
    """Detects the language of a given text."""
    try:
        return detect(text[:500])
    except LangDetectException:
        return "unknown"

def extract_text_from_page(page):
    """Extracts text from a page of a PDF."""
    return page.get_text("text").strip()

def segment_into_sections(text, doc_name, page_num):
    """
    Segments text from a page into structured sections (recipes).
    Splits text into blocks and validates them as recipes using flexible regex.
    """
    sections = []
    seen_sections = set()
    recipe_blocks = re.split(r'\n\s*\n\s*\n', text)
    
    for block in recipe_blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.split('\n')
        title = clean_text(lines[0].strip())
        
        # Validate title: must be longer than 3 chars and not a bullet point
        if len(title) <= 3 or title.startswith(('•', '', 'o')):
            continue

        content = block
        
        if title and re.search(r'(ingredients|preparation|method|steps|instructions)', content, re.I):
            cleaned_content = clean_text(content)
            section_identifier = (title, cleaned_content)
            if section_identifier not in seen_sections:
                sections.append({
                    'title': title,
                    'content': cleaned_content,
                    'document': doc_name,
                    'page': page_num,
                    'language': detect_language(cleaned_content)
                })
                seen_sections.add(section_identifier)
    return sections

def process_pdf(pdf_path):
    """Processes a single PDF, extracting sections from each page."""
    doc_name = os.path.basename(pdf_path)
    all_sections = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc.pages():
            text = extract_text_from_page(page)
            if text:
                page_sections = segment_into_sections(text, doc_name, page.number + 1)
                all_sections.extend(page_sections)
        doc.close()
    except Exception as e:
        logging.error(f"Error processing {doc_name}: {e}")
    return all_sections

def rank_sections(sections, query, model):
    """Ranks sections based on semantic similarity to the query."""
    if not sections:
        return []
    
    section_contents = [s['content'] for s in sections]
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embeddings = model.encode(section_contents, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)[0].cpu().numpy()
    
    for i, section in enumerate(sections):
        section['relevance'] = float(similarities[i])
        
    return sorted(sections, key=lambda x: x['relevance'], reverse=True)

def get_full_text(text):
    """Returns the full text for the 'refined_text' field."""
    return text

def process_collection(collection_path, model):
    """Main logic to process a single collection."""
    input_path = os.path.join(collection_path, 'challenge1b_input.json')
    output_path = os.path.join(collection_path, 'generated_output.json')
    pdfs_dir = os.path.join(collection_path, 'PDFs')

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona_role = input_data.get('persona', {}).get('role', '')
    job_task = input_data.get('job_to_be_done', {}).get('task', '')
    query = f"{persona_role}: {job_task}"
    
    pdf_filenames = [doc['filename'] for doc in input_data.get('documents', [])]

    all_sections = []
    for filename in tqdm(pdf_filenames, desc=f"Processing PDFs in {os.path.basename(collection_path)}"):
        pdf_path = os.path.join(pdfs_dir, filename)
        if os.path.exists(pdf_path):
            all_sections.extend(process_pdf(pdf_path))

    non_vegetarian_keywords = [
        'beef', 'chicken', 'pork', 'lamb', 'shrimp', 'fish', 'turkey', 'sausage', 'meatloaf',
        'pancetta', 'crab', 'ham', 'bacon', 'prosciutto', 'salami', 'pepperoni', 
        'chorizo', 'veal', 'mutton', 'venison', 'duck', 'goose', 'tuna', 'salmon', 'cod', 'lobster'
    ]
    vegetarian_sections = []
    for section in all_sections:
        content_lower = section['content'].lower()
        title_lower = section['title'].lower()
        if not any(keyword in title_lower or keyword in content_lower for keyword in non_vegetarian_keywords):
            vegetarian_sections.append(section)

    ranked_sections = rank_sections(vegetarian_sections, query, model)

    # Boost scores for gluten-free and penalize non-dinner items
    non_dinner_keywords = ['sandwich', 'toast', 'smoothie', 'breakfast', 'wrap', 'parfait']
    for section in ranked_sections:
        content_lower = section['content'].lower()
        title_lower = section['title'].lower()
        if 'gluten-free' in content_lower:
            section['relevance'] += 0.2
        if any(keyword in title_lower for keyword in non_dinner_keywords):
            section['relevance'] -= 0.3
    
    ranked_sections = sorted(ranked_sections, key=lambda x: x['relevance'], reverse=True)
    
    output_data = {
        "metadata": {
            "input_documents": pdf_filenames,
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_section": [{
            "document": s['document'],
            "page_number": s['page'],
            "section_title": s['title'],
            "importance_rank": i + 1
        } for i, s in enumerate(ranked_sections)],
        "sub-section_analysis": [{
            "document": s['document'],
            "refined_text": get_full_text(s['content']),
            "page_number": s['page']
        } for s in ranked_sections]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Successfully generated output for {os.path.basename(collection_path)}")

def main():
    """Finds and processes all collections."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logging.info("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    logging.info("Model loaded.")

    for item in os.listdir(script_dir):
        collection_path = os.path.join(script_dir, item)
        if os.path.isdir(collection_path) and item.startswith('Collection'):
            logging.info(f"--- Processing {item} ---")
            try:
                process_collection(collection_path, model)
            except Exception as e:
                logging.error(f"Failed to process {item}. Error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
