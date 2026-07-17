import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self.collection_name = "knowledge_base"
        
        # Load size dynamically from embeddings service
        from app.services.embeddings import embeddings_service
        self.vector_size = embeddings_service.dimension
        
        self.qdrant_client = None
        self.use_fallback = False
        
        # In-memory database structure for fallback mode
        self.fallback_db: List[Dict[str, Any]] = []

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            logger.info("Initializing in-memory Qdrant database...")
            # Using ":memory:" runs an actual Qdrant engine in RAM locally without Docker!
            self.qdrant_client = QdrantClient(location=":memory:")
            
            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info("Qdrant in-memory collection 'knowledge_base' created successfully.")
        except Exception as e:
            logger.warning(
                f"Could not initialize native QdrantClient ({e}). "
                f"Falling back to Python in-memory vector store."
            )
            self.qdrant_client = None
            self.use_fallback = True

    def add_chunks(self, texts: List[str], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]):
        """Adds text chunks and their embeddings to the vector database."""
        if not texts or not embeddings:
            return

        # Always populate fallback database for maximum search resilience
        for text, vector, meta in zip(texts, embeddings, metadata_list):
            self.fallback_db.append({
                "text": text,
                "vector": vector,
                "metadata": meta
            })
        logger.info(f"Saved {len(texts)} chunks to fallback vector store. Total chunks: {len(self.fallback_db)}")

        if not self.use_fallback and self.qdrant_client:
            try:
                from qdrant_client.models import PointStruct
                points = []
                for idx, (text, vector, meta) in enumerate(zip(texts, embeddings, metadata_list)):
                    # Merge text into metadata
                    payload = {**meta, "text": text}
                    # We can use a hash of the text or index as id
                    point_id = hash(text) % (10**8)
                    points.append(PointStruct(id=point_id, vector=vector, payload=payload))
                
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )
                logger.info(f"Upserted {len(texts)} chunks to Qdrant collection.")
            except Exception as e:
                logger.error(f"Error upserting to Qdrant: {e}. Defaulting searches to fallback vector store.")
                self.use_fallback = True

    def search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Searches the vector database for the top matches similar to query_vector."""
        if not self.use_fallback and self.qdrant_client:
            try:
                results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit
                )
                return [
                    {
                        "text": hit.payload.get("text", ""),
                        "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                        "score": hit.score
                    }
                    for hit in results.points
                ]
            except Exception as e:
                logger.error(f"Error querying Qdrant: {e}. Using fallback vector store.")
                self.use_fallback = True

        # Fallback implementation using cosine similarity
        results = []
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm == 0:
            return []

        for item in self.fallback_db:
            i_vec = np.array(item["vector"])
            i_norm = np.linalg.norm(i_vec)
            
            if i_norm == 0:
                continue
                
            similarity = float(np.dot(q_vec, i_vec) / (q_norm * i_norm))
            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": similarity
            })
            
        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Returns unique filenames of ingested documents."""
        filenames = set()
        
        if not self.use_fallback and self.qdrant_client:
            try:
                # Scroll all points to find document names
                res, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                for point in res:
                    fname = point.payload.get("filename")
                    if fname:
                        filenames.add(fname)
            except Exception as e:
                logger.error(f"Error scrolling Qdrant: {e}")
                
        # Also check fallback DB
        for item in self.fallback_db:
            fname = item["metadata"].get("filename")
            if fname:
                filenames.add(fname)
                
        return [{"filename": name} for name in sorted(list(filenames))]

    def reset(self):
        """Resets the vector database."""
        self.fallback_db = []
        if not self.use_fallback and self.qdrant_client:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                from qdrant_client.models import Distance, VectorParams
                from app.services.embeddings import embeddings_service
                self.vector_size = embeddings_service.dimension
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
            except Exception as e:
                logger.error(f"Error resetting Qdrant: {e}")

# Singleton instance
vector_db = VectorDBService()
