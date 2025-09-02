import logging, os
from src.logging_config import setup_logging
from typing import List, Dict
from .document_processor import DocumentChunk
import chromadb
from chromadb.config import Settings

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Vector database for storing and searching document chunks
    Uses ChromaDB for persistent storage with cosine similarity
    """
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        self.collection_name = "insurance_documents"
        self.collection = None
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or get existing collection with cosine similarity"""
        try:
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized collection: {self.collection_name} with cosine similarity")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add document chunks to the vector database"""
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                'document_source': chunk.document_source,
                'document_title': chunk.document_title,
                'page_number': str(chunk.page_number) if chunk.page_number else '',
                'section_title': chunk.section_title or '',
                'chunk_index': str(chunk.chunk_index),
                **{k: str(v) for k, v in chunk.metadata.items()}
            } for chunk in chunks
        ]

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector database")
        except Exception as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            raise

    def search(self, query_embedding: List[float], n_results: int = 10, threshold: float = 0.65) -> Dict:
        """Search for similar chunks with threshold filter"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"]
            )
            filtered = {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            for i in range(len(results['distances'][0])):
                sim = 1 - results['distances'][0][i]
                if sim >= threshold:
                    filtered['ids'][0].append(results['ids'][0][i])
                    filtered['documents'][0].append(results['documents'][0][i])
                    filtered['metadatas'][0].append(results['metadatas'][0][i])
                    filtered['distances'][0].append(results['distances'][0][i])
                if len(filtered['ids'][0]) >= n_results:
                    break
            return filtered
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name
        }

    def document_exists(self, document_title: str) -> bool:
        """Check if document chunks already exist"""
        results = self.collection.get(where={"document_title": document_title})
        return len(results['ids']) > 0