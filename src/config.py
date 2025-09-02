import os
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    cache_dir: str
    model_name: str

@dataclass
class DatabaseConfig:
    db_path: str

@dataclass
class SynthesisConfig:
    max_output_tokens: int
    temperature: float

@dataclass
class SystemConfig:
    documents_dir: str
    logs_dir: str
    cache_dir: str
    db_path: str
    top_k_results: int
    similarity_threshold: float
    max_chunk_size: int
    chunk_overlap: int
    log_level: str
    embedding: EmbeddingConfig
    database: DatabaseConfig
    synthesis: SynthesisConfig

    @classmethod
    def from_env(cls) -> 'SystemConfig':
        return cls(
            documents_dir=os.getenv("DOCUMENTS_DIR", "./documents"),
            logs_dir=os.getenv("LOGS_DIR", "./logs"),
            cache_dir=os.getenv("CACHE_DIR", "./cache"),
            db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
            top_k_results=int(os.getenv("TOP_K_RESULTS", 10)),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", 0.65)),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1200)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            embedding=EmbeddingConfig(
                cache_dir=os.getenv("CACHE_DIR", "./cache"),
                model_name="all-mpnet-base-v2"
            ),
            database=DatabaseConfig(
                db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
            ),
            synthesis=SynthesisConfig(
                max_output_tokens=2048,
                temperature=0.2
            )
        )

    def validate(self) -> List[str]:
        """Validate configuration attributes"""
        errors = []
        for dir_path in [self.documents_dir, self.logs_dir, self.cache_dir, self.db_path]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {str(e)}")

        if self.top_k_results < 1:
            errors.append(f"top_k_results must be positive, got {self.top_k_results}")
        if not 0 <= self.similarity_threshold <= 1:
            errors.append(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        if self.max_chunk_size < 100:
            errors.append(f"max_chunk_size must be at least 100, got {self.max_chunk_size}")
        if self.chunk_overlap < 0:
            errors.append(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")

        service = os.getenv("SERVICE", "groq").lower()
        if service == "groq" and not os.getenv("GROQ_API_KEY"):
            errors.append("GROQ_API_KEY is required when SERVICE=groq")
        elif service == "gemini" and not os.getenv("GEMINI_API_KEY"):
            errors.append("GEMINI_API_KEY is required when SERVICE=gemini")

        return errors