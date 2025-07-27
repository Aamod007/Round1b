# Approach Explanation

**Model Used:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Core Logic & Methodology

The document intelligence system employs a sophisticated, two-stage pipeline to analyze PDF documents and extract highly relevant, task-focused information.

1.  **Persona and Task Interpretation**: The system begins by creating a semantic query from the `persona` and `job_to_be_done` fields in the input JSON. This query serves as the ground truth for all subsequent relevance scoring.

2.  **Intelligent Sectioning with Layout Analysis**: Each PDF is parsed using **PyMuPDF**, which provides detailed layout information. The script analyzes font attributes (size and weight) to heuristically identify section titles, allowing it to logically segment the document content even without explicit bookmarks. This method is more robust than simple text splitting.

3.  **Multilingual Semantic Ranking (Stage 1)**: The content of each identified section is vectorized using the `paraphrase-multilingual-MiniLM-L12-v2` model. This powerful multilingual model computes the cosine similarity between each section and the persona-driven query. Sections are then ranked by this similarity score, and only those exceeding a **0.4 threshold** are considered for the next stage. The top 5 sections are selected.

4.  **Paragraph-Level Refinement (Stage 2)**: For each of the top-ranked sections, the system performs a second level of analysis. The section's content is broken down into individual paragraphs. Each paragraph is then semantically scored against the same persona query. The 1-2 most relevant paragraphs are extracted and combined to form the final `refined_text`. This ensures the final output is both relevant and concise.

5.  **Multilingual Handling**: The use of a dedicated multilingual sentence transformer ensures that content in all specified languages (English, French, Korean, etc.) is processed and understood natively without the need for a separate translation step, preserving semantic accuracy.

6.  **Final Output Generation**: The top 5 sections and their corresponding refined text snippets are compiled into the `generated_output.json`. A single, consolidated `approach_explanation.md` is also created in the root directory to document this methodology.
