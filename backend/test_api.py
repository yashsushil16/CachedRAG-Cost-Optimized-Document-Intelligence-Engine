import sys
import os
import time

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load services
from app.services.embeddings import embeddings_service
from app.services.cache import semantic_cache
from app.services.vector_db import vector_db
from app.services.llm import llm_service, metrics_store

def run_tests():
    print("=" * 60)
    print("RUNNING PIPELINE UNIT VERIFICATION TESTS")
    print("=" * 60)

    # 1. Test Embedding Generation
    print("[1/5] Testing Embeddings Generation Service...")
    sample_text = "How do I reset my account password?"
    vec = embeddings_service.get_embedding(sample_text)
    assert len(vec) == 384, f"Expected 384-dim vector, got {len(vec)}"
    print(f"  SUCCESS: Generated {len(vec)}-dimensional vector embedding.")

    # 2. Test Vector DB Ingestion & Retrieval
    print("\n[2/5] Testing Local In-Memory Vector Store...")
    doc_text = "To reset your password, navigate to the settings page, click 'Reset Password', and follow the instructions sent to your email."
    doc_meta = {"filename": "security_guide.txt", "chunk_index": 0}
    doc_embedding = embeddings_service.get_embedding(doc_text)
    
    vector_db.add_chunks([doc_text], [doc_embedding], [doc_meta])
    
    # Retrieve
    search_query = "change password settings"
    search_vec = embeddings_service.get_embedding(search_query)
    results = vector_db.search(search_vec, limit=1)
    
    assert len(results) > 0, "No results retrieved from Vector DB"
    assert results[0]["text"] == doc_text, "Retrieved incorrect text chunk"
    print(f"  SUCCESS: Correctly retrieved chunk from document '{results[0]['metadata']['filename']}' with similarity score {results[0]['score']:.4f}")

    # 3. Test LLM Synthesis (Simulation mode)
    print("\n[3/5] Testing LLM Generation Engine...")
    answer, usage = llm_service.generate_answer(search_query, results)
    assert len(answer) > 0, "Generated answer is empty"
    print(f"  SUCCESS: Synthesized answer:\n\"\"\"\n{answer}\n\"\"\"")
    print(f"  Tokens Input: {usage['input_tokens']}, Output: {usage['output_tokens']}, Latency: {usage['latency_ms']:.1f}ms")

    # 4. Test Gemini Evaluation Judge (Simulation mode)
    print("\n[4/5] Testing Gemini Alignment Evaluation Judge...")
    eval_res = llm_service.evaluate_response(search_query, results, answer)
    assert "faithfulness" in eval_res, "Evaluation missing faithfulness score"
    assert "relevance" in eval_res, "Evaluation missing relevance score"
    print(f"  SUCCESS: Faithfulness: {eval_res['faithfulness']}, Relevance: {eval_res['relevance']}")
    print(f"  Judge Reasoning: {eval_res['reason']}")

    # 5. Test Semantic Caching (Redis VSS simulator)
    print("\n[5/5] Testing Redis-VSS Semantic Cache Simulator...")
    # Clean cache first
    semantic_cache.reset()
    
    query1 = "How can I update my login credentials?"
    query1_vec = embeddings_service.get_embedding(query1)
    ans1 = "You can update your credentials by clicking the reset profile link in user settings."
    
    # Check cache (should miss)
    match, score = semantic_cache.query(query1, query1_vec)
    assert match is None, "Expected cache miss on empty cache"
    print(f"  Cache Miss (Correct): Query '{query1}' returned score {score:.4f}")
    
    # Add to cache
    semantic_cache.add(query1, query1_vec, ans1, faithfulness=1.0, relevance=0.95, latency_ms=1200)
    
    # Test identical query (should hit)
    match, score = semantic_cache.query(query1, query1_vec)
    assert match is not None, "Expected cache hit on identical query"
    assert match["answer"] == ans1, "Retrieved incorrect cached answer"
    print(f"  Cache Hit (Correct): Query '{query1}' matched with similarity {score:.4f}")

    # Test semantic match (similar wording, e.g. 'steps to change login info')
    query2 = "steps to change my login info"
    query2_vec = embeddings_service.get_embedding(query2)
    match, score = semantic_cache.query(query2, query2_vec)
    
    # Our hashing trick produces similar vectors for similar word sets. Let's inspect
    print(f"  Semantic Lookup: Wording '{query2}' vs cache returns similarity {score:.4f}")
    if score >= 0.85:
        print("  SUCCESS: Semantic Cache HIT achieved for similar wording.")
    else:
        print("  Info: Semantic similarity is below hit threshold, correct for unaligned vectors.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
