import json
import logging
import time
from threading import Lock
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class SemanticCacheService:
    def __init__(self):
        self.lock = Lock()
        self.cache_file = settings.DATA_DIR / "semantic_cache.json"
        self.entries: List[Dict[str, Any]] = []
        self.hits = 0
        self.misses = 0
        self.load_cache()

    def clean_query_for_cache(self, text: str) -> str:
        """Normalizes query text to maximize semantic cache hit rate."""
        import re
        if not text:
            return ""
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove common conversational fillers and auxiliary words
        fillers = [
            r"\bplease\b", r"\bcould you\b", r"\bcan you\b", r"\btell me\b",
            r"\bshow me\b", r"\bhow do i\b", r"\bhow to\b", r"\bis there a way to\b",
            r"\bsteps to\b", r"\bwhat is\b", r"\bwhat are\b", r"\bgive me\b",
            r"\bfind\b", r"\bsearch for\b", r"\blist of\b", r"\bcan i\b", r"\bdo i\b"
        ]
        for filler in fillers:
            text = re.sub(filler, '', text)
        # Standardize whitespace
        return " ".join(text.split())

    def load_cache(self):
        """Loads cached queries and answers from disk."""
        with self.lock:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, "r") as f:
                        data = json.load(f)
                        self.entries = data.get("entries", [])
                        self.hits = data.get("hits", 0)
                        self.misses = data.get("misses", 0)
                    logger.info(f"Loaded {len(self.entries)} cache entries from {self.cache_file}")
                except Exception as e:
                    logger.error(f"Error loading semantic cache: {e}")
                    self.entries = []
            else:
                self.entries = []

    def save_cache(self):
        """Saves current cache entries and metrics to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump({
                    "entries": self.entries,
                    "hits": self.hits,
                    "misses": self.misses
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving semantic cache: {e}")

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity between two vectors."""
        arr1 = np.array(v1)
        arr2 = np.array(v2)
        
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def query(self, query_text: str, query_vector: List[float]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Queries the semantic cache for a similar vector.
        Returns (match_entry, similarity_score) if a match is found above threshold.
        """
        with self.lock:
            best_match = None
            best_score = -1.0
            
            for entry in self.entries:
                similarity = self.cosine_similarity(query_vector, entry["vector"])
                if similarity > best_score:
                    best_score = similarity
                    best_match = entry
            
            # Check if match meets the similarity threshold
            if best_match and best_score >= settings.SEMANTIC_CACHE_THRESHOLD:
                self.hits += 1
                best_match["hit_count"] = best_match.get("hit_count", 0) + 1
                best_match["last_hit_at"] = time.time()
                self.save_cache()
                logger.info(f"Semantic Cache HIT: '{query_text}' matches '{best_match['query']}' with similarity {best_score:.4f}")
                return best_match, best_score
            
            self.misses += 1
            logger.info(f"Semantic Cache MISS: '{query_text}' (Best similarity: {best_score:.4f})")
            return None, best_score

    def add(self, query_text: str, query_vector: List[float], answer: str, 
            faithfulness: float = 1.0, relevance: float = 1.0, latency_ms: float = 0.0):
        """Adds a new query, its vector, and generated answer to the cache."""
        with self.lock:
            # Check if query already exists (using cleaned text comparison) to prevent duplicate exact strings
            cleaned_target = self.clean_query_for_cache(query_text)
            for entry in self.entries:
                if self.clean_query_for_cache(entry["query"]) == cleaned_target:
                    # Update existing entry
                    entry["answer"] = answer
                    entry["vector"] = query_vector
                    entry["faithfulness"] = faithfulness
                    entry["relevance"] = relevance
                    entry["latency_ms"] = latency_ms
                    entry["created_at"] = time.time()
                    self.save_cache()
                    return
            
            # Insert new entry
            self.entries.append({
                "query": query_text,
                "vector": query_vector,
                "answer": answer,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "latency_ms": latency_ms,
                "hit_count": 0,
                "created_at": time.time(),
                "last_hit_at": None
            })
            
            self.save_cache()
            logger.info(f"Added query '{query_text}' to Semantic Cache.")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns statistics of the cache."""
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_ratio": hit_ratio,
                "cached_items_count": len(self.entries)
            }

    def reset(self):
        """Resets the cache store and metrics."""
        with self.lock:
            self.entries = []
            self.hits = 0
            self.misses = 0
            self.save_cache()
            logger.info("Semantic cache reset completed.")

# Singleton instance
semantic_cache = SemanticCacheService()
