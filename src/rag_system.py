import logging, os
from src.logging_config import setup_logging
from typing import List, Dict, Any
from pathlib import Path
from .document_processor import InsuranceDocumentProcessor, DocumentChunk
from .embedding_service import EmbeddingService
from .vector_database import VectorDatabase
from .query_classifier import QueryClassifier
from .response_synthesizer import ResponseSynthesizer, SynthesisResult, RetrievalResult
from .config import SystemConfig

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class InsuranceRAGSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.document_processor = InsuranceDocumentProcessor(config)
        self.embedding_service = EmbeddingService(model_name=config.embedding.model_name, cache_dir=config.embedding.cache_dir)
        self.vector_db = VectorDatabase(config.database.db_path)
        self.query_classifier = QueryClassifier()
        self.synthesizer = ResponseSynthesizer(config)
        self.documents = []
        self.default_documents = [
            {
                'url': 'https://www.cms.gov/cciio/resources/forms-reports-and-other-resources/downloads/sbc-sample-completed-mm-508-fixed-4-12-16.pdf',
                'title': 'CMS_Completed_Summary_Benefits_Coverage'
            },
            {
                'url': 'https://www.cms.gov/cciio/resources/files/downloads/sbc-sample.pdf',
                'title': 'CMS_Sample_Summary_Benefits_Template'
            }
        ]

    def setup_system(self, download_default_docs: bool = True):
        """Initial system setup, load existing if available"""
        logger.info("Setting up Insurance RAG System...")
        Path(self.config.documents_dir).mkdir(exist_ok=True)
        Path(self.config.logs_dir).mkdir(exist_ok=True)
        Path(self.config.cache_dir).mkdir(exist_ok=True)

        if download_default_docs:
            self._download_default_documents()

        doc_files = list(Path(self.config.documents_dir).glob("*.pdf"))
        if doc_files:
            for doc_file in doc_files:
                doc_title = next(
                    (doc['title'] for doc in self.default_documents if doc['title'] in str(doc_file)),
                    str(doc_file.stem)
                )
                if not self.vector_db.document_exists(doc_title):
                    self.ingest_document(str(doc_file), doc_title)
                else:
                    logger.info(f"Skipping ingestion for already existing document: {doc_title}")
        else:
            logger.warning("No documents found to process")

    def _download_default_documents(self):
        """Download default insurance documents if not present"""
        for doc_info in self.default_documents:
            file_path = Path(self.config.documents_dir) / f"{doc_info['title']}.pdf"
            if not file_path.exists():
                logger.info(f"Downloading {doc_info['title']}...")
                self.document_processor.download_document(doc_info['url'], str(file_path))

    def ingest_document(self, pdf_path: str, document_title: str):
        """Ingest a document and add it to the vector database"""
        try:
            chunks = self.document_processor.process_document(pdf_path, document_title)
            if chunks:
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)
                self.vector_db.add_chunks(chunks, embeddings)
                self.documents.append({'path': pdf_path, 'title': document_title})
                logger.info(f"Ingested document: {pdf_path} with {len(chunks)} chunks")
            else:
                logger.warning(f"No chunks created from {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to ingest document {pdf_path}: {str(e)}")

    def add_new_document(self, url: str, document_title: str):
        """Download and ingest a new document if it doesn't exist"""
        if self.vector_db.document_exists(document_title):
            logger.info(f"Document {document_title} already exists in database, skipping ingestion")
            return
        file_path = Path(self.config.documents_dir) / f"{document_title}.pdf"
        if self.document_processor.download_document(url, str(file_path)):
            self.ingest_document(str(file_path), document_title)

    def query(self, query: str) -> SynthesisResult:
        """Process a user query and return the synthesized answer"""
        try:
            logger.info(f"Processing query: {query}")
            query_type = self.query_classifier.classify_query(query)

            query_embedding = self.embedding_service.get_embedding(query)
            retrieved_chunks = self.vector_db.search(query_embedding, n_results=self.config.top_k_results, threshold=self.config.similarity_threshold)
            
            processed_chunks = []
            for i in range(len(retrieved_chunks['ids'][0])):
                processed_chunks.append(RetrievalResult(
                    id=retrieved_chunks['ids'][0][i],
                    content=retrieved_chunks['documents'][0][i],
                    metadata=retrieved_chunks['metadatas'][0][i],
                    similarity_score=1 - retrieved_chunks['distances'][0][i]
                ))

            if query_type in ['comparative', 'aggregate']:
                processed_chunks = sorted(processed_chunks, key=lambda x: x.similarity_score, reverse=True)
                unique_docs = []
                filtered_chunks = []
                for chunk in processed_chunks:
                    doc_title = chunk.metadata['document_title']
                    if doc_title not in unique_docs or len(unique_docs) < len(self.documents):
                        filtered_chunks.append(chunk)
                        unique_docs.append(doc_title)
                processed_chunks = filtered_chunks[:self.config.top_k_results]

            if not processed_chunks:
                logger.warning(f"No relevant chunks found for query: {query}")
                return SynthesisResult(
                    answer="No relevant information found in the provided documents.",
                    sources=[],
                    confidence_score=0.0,
                    query_type=query_type,
                    relevant_chunks=[]
                )

            result = self.synthesizer.synthesize_answer(query, processed_chunks, query_type)
            logger.info(f"Generated answer with confidence: {result.confidence_score:.2f}")
            return result
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            return SynthesisResult(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                query_type=query_type,
                relevant_chunks=[]
            )

    def get_system_stats(self) -> Dict[str, Any]:
        """Return system statistics"""
        return {
            'total_chunks': self.vector_db.get_collection_stats()['total_chunks'],
            'documents_directory': self.config.documents_dir,
            'documents': [doc['title'] for doc in self.documents]
        }