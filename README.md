# Insurance RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing multiple insurance policy documents, supporting complex queries with source attribution and confidence scores. Built with FastAPI for interactive demo and evaluation, using Groq's llama-3.1-70b-versatile for generation and all-mpnet-base-v2 for embeddings.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Design Documentation](#design-documentation)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Anandtech09/rag_insurance_system.git
cd rag_insurance_system
```

### 2. Run the Setup Script

**For Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**For Windows (run in Command Prompt or PowerShell):**
```bash
./setup.sh
```
or
```bash
bash ./setup.sh
```
This creates a virtual environment, installs dependencies, and sets up directories (`documents/`, `logs/`, `cache/`, `chroma_db/`).

### 3. Configure Environment

Edit the `.env` file to add your Groq API key (preferred) or Gemini API key:

```bash
nano .env
```

Set:
```bash
GROQ_API_KEY=your_groq_api_key_here
SERVICE=groq
DOCUMENTS_DIR=./documents
LOGS_DIR=./logs
CACHE_DIR=./cache
CHROMA_DB_PATH=./chroma_db
TOP_K_RESULTS=10
SIMILARITY_THRESHOLD=0.65
MAX_CHUNK_SIZE=1200
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

If using Gemini:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
SERVICE=gemini
```

### 4. Install Dependencies

Requires Python 3.8+. The `setup.sh` script handles this, but you can manually run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
```

Verify NLTK data:
```bash
python -c "import nltk; print(nltk.data.find('tokenizers/punkt'))"
```

Set PYTHONPATH for module imports:

**Windows:**
```bash
set PYTHONPATH=%PYTHONPATH%;C:\path\to\rag_insurance_system
```

**Linux/macOS:**
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/rag_insurance_system
```

### 5. Run the System

#### Full System Test

Downloads two CMS insurance PDFs (if not already in `documents/`), processes them into chunks, generates embeddings, stores them in ChromaDB

```bash
python scripts/run_system.py
```

#### Evaluate through custom queries
Evaluate through custom queries, outputting answers with source citations and confidence scores.
The queries provided are the Test questions provided in requirement

```bash
python scripts/evaluate_system.py
```

#### Interactive Demo (FastAPI)

Starts a web server for interactive querying and document addition.

```bash
python scripts/demo.py
```

Access at http://localhost:8000/docs for Swagger UI. 

**Endpoints:**
- `GET /stats`: System statistics (e.g., total chunks, documents)
- `POST /query`: Submit a query (body: `{"query": "your question"}`)
- `POST /add-document`: Add a new document (body: `{"url": "document_url", "document_title": "title"}`)

#### Comprehensive Evaluation (FastAPI)

Evaluates queries for retrieval quality, synthesis accuracy, source attribution, query understanding, and completeness. Saves a report to `evaluation_report_[timestamp].json`.

```bash
python scripts/test_system.py
```

Access at http://localhost:8001/docs. 

**Endpoint:**
- `POST /evaluate`: Evaluate queries (body: `{"queries": [{"query": "question", "expected_aspects": ["aspect1", "aspect2"]}]}`)

#### Unit Tests

Tests document processing, embedding, vector database, query classification, and response synthesis.

```bash
python -m pytest tests/test_rag_system.py
```

## Usage

### Documents

Place insurance PDFs in `documents/`, or use default CMS documents (downloaded automatically):
- https://www.cms.gov/cciio/resources/files/downloads/sbc-sample.pdf
- https://www.cms.gov/cciio/resources/forms-reports-and-other-resources/downloads/sbc-sample-completed-mm-508-fixed-4-12-16.pdf

### Adding New Documents

Use the `/add-document` endpoint or:

```python
system.add_new_document("https://new-url.pdf", "New_Document_Title")
```

Existing documents are not re-chunked (checked via `document_title` in ChromaDB).

### Queries

Example queries:
- "What is the deductible for in-network services?"
- "Compare prescription drug coverage between plans."
- "What services are not covered?"

**Output:** Answers include specific details, source citations (document title, page, section), and confidence scores.

### Example Output

QUERY: What is the difference in deductibles between in-network and out-of-network services?
Query Type: COMPARATIVE
Confidence Score: 0.95

ANSWER:
For the CMS Completed Summary of Benefits Coverage, the in-network deductible is $1,500 (individual) and $3,000 (family), while the out-of-network deductible is $3,000 (individual) and $6,000 (family) [Source 1: CMS_Completed_Summary_Benefits_Coverage, Page 1, Section: Important Questions].

For the CMS Sample Summary of Benefits Template, the in-network deductible is $2,000 (individual) and $4,000 (family), while the out-of-network deductible is $4,000 (individual) and $8,000 (family) [Source 2: CMS_Sample_Summary_Benefits_Template, Page 1, Section: Important Questions].

SOURCES:
- [Source 1: CMS_Completed_Summary_Benefits_Coverage, Page 1, Section: Important Questions]
- [Source 2: CMS_Sample_Summary_Benefits_Template, Page 1, Section: Important Questions]

## Design Documentation

### Directory Structure

```
insurance-rag-system/
├── logs/
├── cache/
├── documents/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── query_classifier.py
│   ├── vector_database.py
│   ├── response_synthesizer.py
│   ├── rag_system.py
|   ├── logging_config.py
├── scripts/
│   ├── __init__.py
│   ├── run_system.py
│   ├── demo.py
│   ├── test_system.py
|   ├── evaluate_system.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluation_metrics.py
├── requirements.txt
├── setup.sh
├── .env
└── README.md
```

### Retrieval Approach

- **Document Processing:** PDFs parsed using PyMuPDF for text and tables (converted to Markdown). spaCy identifies sections (e.g., "Deductibles," "Coverage").
- **Embedding:** Uses all-mpnet-base-v2 for high-quality semantic embeddings, cached for efficiency.
- **Vector Database:** ChromaDB with cosine similarity for fast, confident retrieval (similarity_threshold=0.65).
- **Search:** Retrieves top-k chunks, with reranking for comparative/aggregate queries to ensure document diversity.

### Multi-Document Synthesis

- Combines results from multiple documents, prioritizing high-similarity chunks.
- Groq's llama-3.1-70b-versatile (or Gemini's gemini-1.5-pro) synthesizes answers tailored to query type (comparative, aggregate, conditional, gap analysis, specific).
- Table data preserved in Markdown for accurate cost responses (e.g., deductibles, copays).

## Evaluation Metrics

- **Retrieval Quality (30%):** Measures average similarity, document diversity, and relevance (chunks with similarity > 0.7).
- **Synthesis Accuracy (25%):** Evaluates use of insurance terms (e.g., "deductible," "coinsurance"), specific values ($, %), and answer length.
- **Source Attribution (20%):** Checks for proper citations (document title, page, section).
- **Query Understanding (15%):** Assesses query type accuracy and answer relevance to query keywords.
- **Completeness (10%):** Ensures expected aspects are covered, with notes on missing information.

Reports are saved as `evaluation_report_[timestamp].json`.

## Limitations

- **Table Extraction:** Handles structured tables well but may struggle with complex or scanned PDFs (OCR not implemented).
- **Rate Limits:** Groq has higher limits (30/min); Gemini requires delays (10/min) with exponential backoff.
- **Embedding Model:** all-mpnet-base-v2 is robust but could be upgraded to larger models for better semantic understanding.
- **Regulatory Compliance:** Not HIPAA-compliant; sensitive data handling needs enhancement.
- **Scalability:** Tested with small document sets; large-scale performance may require optimization.

## Future Improvements

- Integrate advanced embedding models (e.g., larger transformer models).
- Add OCR for scanned PDFs to improve table extraction.
- Implement query result caching for faster responses.
- Add encryption and access controls for regulatory compliance.
- Optimize for large document sets with parallel processing and batch embedding.

## Troubleshooting

- **API Key Error:** Ensure `GROQ_API_KEY` or `GEMINI_API_KEY` is set in `.env`.
- **Document Download Issue:** Check internet connection or manually place PDFs in `documents/`. URLs may be blocked by some networks.
- **Dependency Problems:** Re-run `pip install -r requirements.txt` and verify Python version (3.8+).
- **FastAPI Access:** Ensure http://localhost:8000 (demo) or http://localhost:8001 (evaluation using query and aspects) is accessible; check firewall settings.
- **Test Failures:** Confirm spaCy (`en_core_web_sm`) and NLTK (`punkt`, `stopwords`) are downloaded.

### For Detailed System Evaluation

Run:
```bash
python scripts/test_system.py
```

Access http://localhost:8001/docs and use the `/evaluate` endpoint with a JSON payload:

```json
{
  "queries": [
    {
      "query": "What is the difference in deductibles between in-network and out-of-network services?",
      "expected_aspects": ["in-network deductible", "out-of-network deductible", "difference"]
    }
  ]
}
```

This generates a comprehensive test report (`evaluation_report_*.json`) with metrics for retrieval, synthesis, attribution, understanding, and completeness.