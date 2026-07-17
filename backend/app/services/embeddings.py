import numpy as np
import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self):
        self.model = None
        self.gemini_enabled = False
        self.initialized = False
        self.dimension = 384  # Default, but updated dynamically
        
        # 1. Try initializing Gemini API first if configured
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_enabled = True
                self.dimension = 768  # Gemini text-embedding-004 uses 768 dimensions
                logger.info("Google Gemini Embeddings service configured (text-embedding-004).")
            except Exception as e:
                logger.warning(f"Could not configure Gemini API for embeddings: {e}. Trying local SentenceTransformers.")

        # 2. Try loading local sentence-transformers if Gemini is disabled/unconfigured
        if not self.gemini_enabled:
            self._init_local_model()
            
        # 3. If neither works, set default dimension based on selected model string
        if not self.gemini_enabled and not self.initialized:
            self.dimension = 768 if "mpnet" in settings.EMBEDDING_MODEL else 384
            logger.info(f"Using deterministic feature hashing fallback. Dimension: {self.dimension}")

    def _init_local_model(self):
        """Initializes local SentenceTransformer model if not already initialized."""
        if self.initialized and self.model:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                self.dimension = 768 if "mpnet" in settings.EMBEDDING_MODEL else 384
            self.initialized = True
            logger.info(f"SentenceTransformer model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer ({e}). "
                f"Falling back to deterministic feature-hashing embedding generator."
            )
            self.model = None
            self.initialized = False

    def get_embedding(self, text: str) -> List[float]:
        """Generates a vector embedding for the input text."""
        if not text:
            return [0.0] * self.dimension

        # Try Gemini API if enabled
        if self.gemini_enabled:
            try:
                import google.generativeai as genai
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query"
                )
                return result["embedding"]
            except Exception as e:
                logger.error(f"Error generating embedding with Gemini API: {e}. Falling back to local model.")
                # Temporarily disable Gemini so we don't spam requests and trigger fallbacks immediately next time
                self.gemini_enabled = False
                self._init_local_model()

        # Try local model
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
            
        # Try Gemini API if enabled
        if self.gemini_enabled:
            try:
                import google.generativeai as genai
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                # Google API returns a list of dictionaries/floats for batch requests
                return result["embedding"]
            except Exception as e:
                logger.error(f"Error generating batch embeddings with Gemini API: {e}. Falling back to local model.")
                self.gemini_enabled = False
                self._init_local_model()

        # Try local model
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
