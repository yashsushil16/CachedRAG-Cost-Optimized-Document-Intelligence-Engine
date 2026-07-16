import numpy as np
import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self):
        self.model = None
        self.dimension = 384  # Standard dimension for all-MiniLM-L6-v2
        self.initialized = False
        
        # Try loading sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.initialized = True
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer ({e}). "
                f"Falling back to deterministic feature-hashing embedding generator."
            )
            self.model = None
            self.initialized = False

    def get_embedding(self, text: str) -> List[float]:
        """Generates a 384-dimensional vector embedding for the input text."""
        if not text:
            return [0.0] * self.dimension

        if self.initialized and self.model:
            try:
                embedding = self.model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding with SentenceTransformer: {e}. Using fallback.")
        
        return self._generate_fallback_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        if not texts:
            return []
            
        if self.initialized and self.model:
            try:
                embeddings = self.model.encode(texts)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}. Using fallback.")
                
        return [self._generate_fallback_embedding(t) for t in texts]

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Deterministic fallback using the hashing trick to generate normalized vectors.
        This ensures cosine similarity remains meaningful (e.g. similar words -> similar vectors)
        without requiring external downloads or heavy libraries.
        """
        import hashlib
        
        # Clean text and split into words
        words = text.lower().strip().split()
        if not words:
            return [0.0] * self.dimension
            
        vector = np.zeros(self.dimension)
        
        # Hash each word to a index and add a contribution
        for word in words:
            # Use md5 for deterministic hashing
            h = hashlib.md5(word.encode('utf-8')).hexdigest()
            # Convert hash parts to dimensions
            idx1 = int(h[:8], 16) % self.dimension
            idx2 = int(h[8:16], 16) % self.dimension
            # Use sign logic for hashing trick (reduces bias)
            sign1 = 1 if int(h[16:20], 16) % 2 == 0 else -1
            sign2 = 1 if int(h[20:24], 16) % 2 == 0 else -1
            
            vector[idx1] += sign1
            vector[idx2] += sign2 * 0.5
            
        # Normalize the vector to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()

# Singleton instance
embeddings_service = EmbeddingsService()
