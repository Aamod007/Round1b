# Document Intelligence System

This repository implements a robust document intelligence pipeline designed to extract highly relevant, task-focused information from PDF documents using advanced NLP models and layout analysis. The system is optimized for multilingual processing and can adapt to a variety of personas and information retrieval tasks.

## Features

- **Multilingual Support:** Processes documents in multiple languages including English, French, Korean, and more.
- **Semantic Search:** Utilizes state-of-the-art sentence transformers for semantic ranking and retrieval.
- **Layout-Aware Sectioning:** Leverages font and layout cues to intelligently segment documents.
- **Two-Stage Extraction:** Performs initial section ranking followed by paragraph-level refinement to pinpoint the most relevant content.
- **User-Centric Queries:** Tailors extraction based on persona and job-to-be-done specifications provided in JSON input.

## How It Works

1. **Semantic Query Generation:** The system constructs a semantic query from the input persona and task description.
2. **PDF Parsing & Sectioning:** PDFs are parsed using PyMuPDF to extract sections based on layout and font characteristics.
3. **Multilingual Embedding & Ranking:** Each section is embedded using the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model and ranked against the semantic query.
4. **Paragraph-Level Refinement:** Top sections are further analyzed at the paragraph level for increased precision.
5. **Output Generation:** The final results, including the top-ranked sections and refined text snippets, are compiled into a JSON file for downstream use.

## Getting Started

### Prerequisites

- Python 3.8+
- `PyMuPDF`
- `sentence-transformers`
- `torch`
- `numpy`
- `pandas` (optional, for data management)

### Installation

```bash
pip install pymupdf sentence-transformers torch numpy pandas
```

### Usage

1. Prepare an input JSON file specifying the `persona` and `job_to_be_done`.
2. Place PDF files to be processed in the designated directory.
3. Run the main pipeline script:

```bash
python main.py --input input.json --pdf_dir ./pdfs --output generated_output.json
```

4. The system will generate `generated_output.json` and a consolidated `approach_explanation.md`.

## Contributing

Contributions and improvements are welcome. Please open issues or submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

---

For a detailed explanation of the underlying methodology, see [Challenge_1b/approach_explanation.md](Challenge_1b/approach_explanation.md).
