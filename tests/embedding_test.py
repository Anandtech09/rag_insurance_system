import logging
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    documents_dir: str
    logs_dir: str
    cache_dir: str
    db_path: str
    synthesis: 'SynthesisConfig'

@dataclass
class SynthesisConfig:
    max_output_tokens: int
    temperature: float

@dataclass
class RetrievalResult:
    id: str
    content: str
    metadata: Dict
    similarity_score: float

@dataclass
class SynthesisResult:
    answer: str
    confidence: float
    sources: List[RetrievalResult]

@dataclass
class DocumentChunk:
    id: str
    content: str
    document_source: str
    document_title: str
    page_number: int
    section_title: str
    chunk_index: int
    metadata: Dict

class EmbeddingService:
    def __init__(self, model_name: str, cache_dir: str):
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Loaded pretrained SentenceTransformer: {model_name}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {str(e)}")
            return []

class VectorDatabase:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="insurance_documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Initialized collection: insurance_documents with cosine similarity")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Added {len(chunks)} chunks to vector database")

    def document_exists(self, document_title: str) -> bool:
        try:
            result = self.collection.get(where={"document_title": document_title})
            logger.info(f"Document exists check for {document_title}: result={result}")
            return len(result.get('ids', [])) > 0
        except Exception as e:
            logger.error(f"Error checking document existence for {document_title}: {str(e)}")
            return False

    def query(self, embedding: List[float], n_results: int = 5) -> List[RetrievalResult]:
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        retrieval_results = []
        for i in range(len(results['ids'][0])):
            retrieval_results.append(RetrievalResult(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity_score=1 - results['distances'][0][i]
            ))
        return retrieval_results

class QueryClassifier:
    def classify_query(self, query: str) -> str:
        query = query.lower().strip()
        if query.startswith("compare") or " vs " in query:
            logger.info(f"Classified query as comparative: {query}")
            return "comparative"
        elif any(word in query for word in ["total", "maximum", "aggregate"]):
            logger.info(f"Classified query as aggregate: {query}")
            return "aggregate"
        elif query.startswith("if") or " in case " in query:
            logger.info(f"Classified query as conditional: {query}")
            return "conditional"
        else:
            logger.info(f"Classified query as general: {query}")
            return "general"

class ResponseSynthesizer:
    def __init__(self, config: SystemConfig, api_key: Optional[str] = None, client: Optional[Groq] = None):
        self.config = config
        if client is not None:
            self.client = client
            logger.info("Initialized ResponseSynthesizer with provided Groq client")
        else:
            try:
                self.client = Groq(api_key=api_key)
                logger.info("Initialized ResponseSynthesizer with service: GROQ")
                logger.info("Groq client initialized with model: llama-3.3-70b-versatile")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                raise

    def _build_prompt(self, query: str, chunks: List[RetrievalResult], query_type: str) -> str:
        prompt = f"You are an expert in analyzing insurance policies. Answer the {query_type} question: {query}\nContext:\n"
        for i, chunk in enumerate(chunks, 1):
            prompt += f"Source {i}: {chunk.metadata['document_title']}, Page {chunk.metadata['page_number']}, Section: {chunk.metadata['section_title']}\n{chunk.content}\n\n"
        prompt += "Provide a precise answer with specific details from the sources, using source references in the format [Source N: Document Title, Page X, Section Y]. If the information is not available, state so explicitly."
        return prompt

    def synthesize_answer(self, query: str, chunks: List[RetrievalResult], query_type: str) -> SynthesisResult:
        logger.info(f"Synthesizing answer using GROQ for query: {query}...")
        prompt = self._build_prompt(query, chunks, query_type)
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=self.config.synthesis.max_output_tokens,
                temperature=self.config.synthesis.temperature
            )
            answer = response.choices[0].message.content
            logger.info("Successfully generated answer using GROQ")
            logger.info(f"Generated answer with confidence: 0.90 using GROQ")
            return SynthesisResult(
                answer=answer,
                confidence=0.90,
                sources=chunks
            )
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {str(e)}")
            return SynthesisResult(
                answer="Failed to generate answer.",
                confidence=0.0,
                sources=chunks
            )

class InsuranceRAGSystem:
    def __init__(self, config: SystemConfig, api_key: Optional[str] = None, client: Optional[Groq] = None):
        self.config = config
        self.embedding_service = EmbeddingService(model_name="all-mpnet-base-v2", cache_dir=config.cache_dir)
        self.vector_db = VectorDatabase(db_path=config.db_path)
        self.query_classifier = QueryClassifier()
        self.response_synthesizer = ResponseSynthesizer(config, api_key=api_key, client=client)

    def query(self, query_text: str) -> SynthesisResult:
        logger.info(f"Processing query: {query_text}")
        query_type = self.query_classifier.classify_query(query_text)
        embedding = self.embedding_service.get_embedding(query_text)
        if not embedding:
            return SynthesisResult(
                answer="Failed to generate embedding for the query.",
                confidence=0.0,
                sources=[]
            )
        chunks = self.vector_db.query(embedding)
        result = self.response_synthesizer.synthesize_answer(query_text, chunks, query_type)
        logger.info(f"Generated answer with confidence: {result.confidence}")
        return result