import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from tqdm import tqdm
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def ocr_image(image_bytes, lang='eng'):
    """
    Performs OCR on an image using pytesseract.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def extract_text_from_page(page, lang='eng'):
    """
    Extracts text from a single PDF page.
    Tries to get native text first, if not available, performs OCR.
    """
    try:
        # Attempt to extract native text
        text = page.get_text("text")
        if text.strip():
            return text.strip()

        # If no native text, perform OCR
        logging.info(f"Page {page.number + 1}: No native text found, performing OCR.")
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        return ocr_image(img_bytes, lang=lang)

    except Exception as e:
        logging.error(f"Error processing page {page.number + 1}: {e}")
        return ""

def process_pdf(pdf_path, languages='eng'):
    """
    Processes a PDF file to extract text from all pages.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return None

    document_name = os.path.basename(pdf_path)
    content = []

    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Processing {document_name} with {doc.page_count} pages.")

        for page_num in tqdm(range(doc.page_count), desc=f"Processing {document_name}"):
            page = doc.load_page(page_num)
            text = extract_text_from_page(page, lang=languages)
            content.append({
                "page": page_num + 1,
                "text": text
            })

        doc.close()
        return {
            "document_name": document_name,
            "content": content
        }

    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_path}: {e}")
        return None

def save_output(data, output_dir, input_filename):
    """
    Saves the extracted data to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f"output_{os.path.splitext(input_filename)[0]}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Output saved to {output_path}")

def main():
    """
    Main function to run the PDF processing script.
    """
    parser = argparse.ArgumentParser(description="Extract text from PDF files using PyMuPDF and Tesseract OCR.")
    parser.add_argument("input_path", help="Path to a single PDF file or a directory containing PDF files.")
    parser.add_argument("-o", "--output_dir", default="Adobe-India-Hackathon25-main/Challenge_1b/output", help="Directory to save the JSON output. Defaults to 'output'.")
    parser.add_argument("-l", "--languages", default="eng+hin+tam+kan+tel", help="Languages for OCR, separated by '+'. Defaults to 'eng+hin+tam+kan+tel'.")
    args = parser.parse_args()

    # --- Script Execution ---
    if not os.path.exists(args.input_path):
        logging.error(f"Input path not found: {args.input_path}")
        return

    if os.path.isdir(args.input_path):
        pdf_files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.info(f"No PDF files found in {args.input_path}")
            return
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.input_path, pdf_file)
            extracted_data = process_pdf(pdf_path, languages=args.languages)
            if extracted_data:
                save_output(extracted_data, args.output_dir, pdf_file)
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith('.pdf'):
        pdf_file = os.path.basename(args.input_path)
        extracted_data = process_pdf(args.input_path, languages=args.languages)
        if extracted_data:
            save_output(extracted_data, args.output_dir, pdf_file)
    else:
        logging.error(f"Invalid input path. Must be a PDF file or a directory of PDF files: {args.input_path}")

if __name__ == '__main__':
    # For Tesseract to work, you might need to specify the path to the executable
    # pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
    # Example for Windows:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main()
