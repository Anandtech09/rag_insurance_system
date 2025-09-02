import logging, os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import pdfplumber
from nltk.tokenize import sent_tokenize
import pandas as pd
import spacy
from src.logging_config import setup_logging
from .config import SystemConfig
try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    logging.warning("tabulate not installed. Tables will be exported as CSV strings instead of Markdown.")


setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    id: str
    content: str
    document_source: str
    document_title: str
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_index: int
    metadata: Dict[str, Any]

class InsuranceDocumentProcessor:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.supported_extensions = {'.pdf'}
        self.nlp = spacy.load("en_core_web_sm")

    def download_document(self, url: str, save_path: str) -> bool:
        """Download document from URL with retry mechanism"""
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded document from {url} to {save_path}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to download {url}: {str(e)}")
        logger.error(f"Failed to download {url} after 3 attempts")
        return False

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text and tables from PDF using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as doc:
                pages_content = []
                for page_num, page in enumerate(doc.pages):
                    text = page.extract_text() or ""
                    table_contents = []
                    tables_found = 0
                    for table_idx, table in enumerate(page.extract_tables()):
                        tables_found += 1
                        df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                        try:
                            if TABULATE_AVAILABLE:
                                table_str = df.to_markdown(index=False)
                            else:
                                table_str = df.to_csv(index=False)
                        except Exception as e:
                            logger.warning(f"Failed to export table to string on page {page_num + 1}: {str(e)}. Using CSV fallback.")
                            table_str = df.to_csv(index=False)
                        table_contents.append(f"\nTable {tables_found}:\n{table_str}\n")
                        logger.info(f"Extracted table {tables_found} on page {page_num + 1} with {len(df)} rows")

                    table_content = "\n\n".join(table_contents)
                    full_content = text + table_content
                    pages_content.append({
                        'page_number': page_num + 1,
                        'content': full_content.strip(),
                        'metadata': {
                            'page_width': page.width,
                            'page_height': page.height
                        }
                    })
                    if tables_found > 0:
                        logger.info(f"Appended {tables_found} tables to page {page_num + 1} content")

            logger.info(f"Extracted {len(pages_content)} pages from {pdf_path} with tables included")
            return pages_content
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            logger.info("Falling back to plain text extraction without tables")
            try:
                with pdfplumber.open(pdf_path) as doc:
                    pages_content = []
                    for page_num, page in enumerate(doc.pages):
                        text = page.extract_text() or ""
                        pages_content.append({
                            'page_number': page_num + 1,
                            'content': text.strip(),
                            'metadata': {'page_width': page.width, 'page_height': page.height}
                        })
                return pages_content
            except Exception as fallback_e:
                logger.error(f"Fallback extraction also failed: {str(fallback_e)}")
                return []

    def identify_sections(self, text: str) -> List[Dict[str, str]]:
        """Identify sections using spaCy for better accuracy"""
        sections = []
        doc = self.nlp(text)
        current_section = None
        section_content = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            is_header = (sent_text.isupper() or
                         any(keyword in sent_text.lower() for keyword in
                             ['coverage', 'benefits', 'deductible', 'copay', 'coinsurance',
                              'limits', 'services', 'costs', 'in-network', 'out-of-network']))
            if is_header:
                if current_section and section_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(section_content)
                    })
                current_section = sent_text
                section_content = []
            elif current_section:
                section_content.append(sent_text)
        if current_section and section_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(section_content)
            })
        if not sections:
            sections.append({
                'title': "Document Content",
                'content': text
            })
        return sections

    def intelligent_chunking(self, text: str, max_chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """Create chunks with sentence boundary preservation, adjusted for tables"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if "| ---" in sentence and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-max(1, overlap//20):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            elif current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-max(1, overlap//20):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def process_document(self, pdf_path: str, document_title: str) -> List[DocumentChunk]:
        """Process a document and return a list of chunks"""
        try:
            pages = self.extract_text_from_pdf(pdf_path)
            chunks = []
            for page in pages:
                content = page['content']
                page_number = page['page_number']
                metadata = page['metadata']
                metadata['document_title'] = document_title
                sections = self.identify_sections(content)
                for section_idx, section in enumerate(sections):
                    section_content = section['content']
                    section_title = section['title']
                    chunk_index = 0
                    chunk_texts = self.intelligent_chunking(
                        section_content,
                        max_chunk_size=self.config.max_chunk_size,
                        overlap=self.config.chunk_overlap
                    )
                    for chunk_text in chunk_texts:
                        chunk_id = f"{document_title}_{page_number}_{section_idx}_{chunk_index}"
                        chunks.append(DocumentChunk(
                            id=chunk_id,
                            content=chunk_text.strip(),
                            document_source=pdf_path,
                            document_title=document_title,
                            page_number=page_number,
                            section_title=section_title,
                            chunk_index=chunk_index,
                            metadata={**metadata, 'section_index': section_title}
                        ))
                        chunk_index += 1
            logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to process document {pdf_path}: {str(e)}")
            return []