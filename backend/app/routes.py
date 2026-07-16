import time
import logging
import io
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.config import settings
from app.services.embeddings import embeddings_service
from app.services.cache import semantic_cache
from app.services.vector_db import vector_db
from app.services.llm import llm_service, metrics_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source: str  # "cache" or "llm"
    cache_similarity: float
    latency_ms: float
    retrieved_chunks: List[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]] = None

def parse_pdf(file_bytes: bytes) -> str:
    """Parses a PDF file from bytes using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {str(e)}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Splits text into chunks of specified size and overlap."""
    if not text:
        return []
    
    # Standardize whitespace
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads, parses, chunks, and indexes a PDF or TXT file."""
    filename = file.filename
    if not filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    try:
        content_bytes = await file.read()
        
        if filename.endswith(".pdf"):
            text = parse_pdf(content_bytes)
        else:
            text = content_bytes.decode("utf-8", errors="ignore")
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded document is empty.")
            
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text chunks.")
            
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks from {filename}...")
        embeddings = embeddings_service.get_embeddings(chunks)
        
        # Build metadata list
        metadata_list = [
            {
                "filename": filename,
                "chunk_index": idx,
                "timestamp": time.time()
            }
            for idx in range(len(chunks))
        ]
        
        # Upsert chunks into Vector DB
        vector_db.add_chunks(chunks, embeddings, metadata_list)
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_count": len(chunks),
            "message": f"Successfully indexed {len(chunks)} text chunks."
        }
    except Exception as e:
        logger.error(f"Error processing document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: QueryRequest, background_tasks: BackgroundTasks):
    """Processes chat queries using Semantic Cache -> RAG Vector DB -> Groq synthesis -> Gemini eval."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    metrics_store.total_queries += 1
    start_time = time.time()
    
    # 1. Generate query embedding
    query_vector = embeddings_service.get_embedding(query)
    
    # 2. Check Semantic Cache (Redis VSS simulation)
    cache_entry, similarity = semantic_cache.query(query, query_vector)
    
    if cache_entry:
        # Cache Hit! Return immediately
        latency_ms = (time.time() - start_time) * 1000
        metrics_store.total_cache_hits += 1
        
        # Log cache savings
        query_words = len(query.split())
        answer_words = len(cache_entry["answer"].split())
        metrics_store.log_cache_savings(query_words + 200, answer_words) # Est tokens
        
        return ChatResponse(
            answer=cache_entry["answer"],
            source="cache",
            cache_similarity=similarity,
            latency_ms=latency_ms,
            retrieved_chunks=[],
            evaluation={
                "faithfulness": cache_entry.get("faithfulness", 1.0),
                "relevance": cache_entry.get("relevance", 1.0),
                "reason": f"Semantic Cache Hit. Similarity: {similarity:.4f}. Reason: {cache_entry.get('reason', 'Retrieved from cache.')}"
            }
        )
    
    # Cache Miss! Go to RAG Pipeline
    metrics_store.total_cache_misses += 1
    
    # 3. Retrieve relevant context from Qdrant Vector DB
    context_chunks = vector_db.search(query_vector, limit=3)
    
    # 4. Generate answer using Primary LLM (Groq)
    answer, primary_usage = llm_service.generate_answer(query, context_chunks)
    
    # 5. Determine if we should evaluate (Dual-LLM validation rate check)
    evaluation_result = None
    if settings.EVALUATION_RATE >= 1.0 or time.time() % 1.0 <= settings.EVALUATION_RATE:
        # Run evaluation judge (Gemini 2.5 Flash)
        evaluation_result = llm_service.evaluate_response(query, context_chunks, answer)
        
        # Log in metrics store
        log_entry = {
            "query": query,
            "answer": answer,
            "faithfulness": evaluation_result["faithfulness"],
            "relevance": evaluation_result["relevance"],
            "reason": evaluation_result["reason"],
            "timestamp": time.time()
        }
        metrics_store.add_eval_log(log_entry)
        
    latency_ms = (time.time() - start_time) * 1000
    
    # 6. Update Semantic Cache asynchronously in background tasks
    faith_val = evaluation_result["faithfulness"] if evaluation_result else 1.0
    relev_val = evaluation_result["relevance"] if evaluation_result else 1.0
    reason_val = evaluation_result["reason"] if evaluation_result else "RAG Generation"
    
    background_tasks.add_task(
        semantic_cache.add,
        query_text=query,
        query_vector=query_vector,
        answer=answer,
        faithfulness=faith_val,
        relevance=relev_val,
        latency_ms=latency_ms
    )
    
    return ChatResponse(
        answer=answer,
        source="llm",
        cache_similarity=similarity,
        latency_ms=latency_ms,
        retrieved_chunks=[
            {
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename", "unknown"),
                "score": chunk["score"]
            }
            for chunk in context_chunks
        ],
        evaluation=evaluation_result
    )

@router.get("/documents")
async def list_documents():
    """Lists all uploaded/indexed documents."""
    return vector_db.get_document_list()

@router.get("/metrics")
async def get_metrics():
    """Returns analytics dashboard metrics."""
    cache_stats = semantic_cache.get_metrics()
    
    # Compute active systems info
    using_mocks = settings.is_simulation_mode
    
    return {
        "summary": {
            "total_queries": metrics_store.total_queries,
            "cache_hits": metrics_store.total_cache_hits,
            "cache_misses": metrics_store.total_cache_misses,
            "cache_hit_ratio": cache_stats["hit_ratio"],
            "cached_items": cache_stats["cached_items_count"]
        },
        "llm_usage": {
            "primary_input_tokens": metrics_store.primary_tokens_input,
            "primary_output_tokens": metrics_store.primary_tokens_output,
            "evaluator_input_tokens": metrics_store.evaluator_tokens_input,
            "evaluator_output_tokens": metrics_store.evaluator_tokens_output,
            "total_cost_usd": round(metrics_store.total_cost_usd, 6),
            "saved_cost_usd": round(metrics_store.saved_cost_usd, 6)
        },
        "evaluation_metrics": {
            "evaluations_count": metrics_store.evaluations_count,
            "average_faithfulness": round(metrics_store.avg_faithfulness, 2),
            "average_relevance": round(metrics_store.avg_relevance, 2),
            "logs": metrics_store.eval_logs[::-1][:20]  # Return last 20 evaluation logs, newest first
        },
        "system_status": {
            "simulation_mode": using_mocks,
            "groq_active": settings.GROQ_API_KEY != "",
            "gemini_active": settings.GEMINI_API_KEY != ""
        }
    }

@router.post("/metrics/reset")
async def reset_metrics():
    """Resets the cache, metrics logs, and indexes."""
    semantic_cache.reset()
    vector_db.reset()
    metrics_store.reset()
    return {"status": "success", "message": "All databases, caches, and metric counters have been reset."}
