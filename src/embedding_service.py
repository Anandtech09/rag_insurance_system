import logging
from src.logging_config import setup_logging
from typing import List
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-mpnet-base-v2", cache_dir: str = "./cache"):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, text: str) -> str:
        """Generate cache file path based on text hash"""
        return str(self.cache_dir / f"{hash(text) % 1000000}.pkl")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with caching"""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        try:
            embedding = self.model.encode(text).tolist()
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with caching"""
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    embeddings[i] = pickle.load(f)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            try:
                new_embeddings = self.model.encode(texts_to_embed, normalize_embeddings=True, batch_size=32)
                for idx, emb in zip(indices_to_embed, new_embeddings):
                    embeddings[idx] = emb.tolist()
                    with open(self._get_cache_path(texts[idx]), 'wb') as f:
                        pickle.dump(emb.tolist(), f)
            except Exception as e:
                logger.error(f"Failed to get batch embeddings: {str(e)}")
                return [[]] * len(texts)

        return embeddings