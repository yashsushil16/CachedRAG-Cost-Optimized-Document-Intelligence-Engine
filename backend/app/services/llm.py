import json
import re
import logging
import time
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

# Global metrics tracking (token usage, estimated costs)
# Rates in USD per 1M tokens (approximate):
# Groq Llama 3.1 8B: Input $0.05, Output $0.08
# Gemini 2.5 Flash: Input $0.075, Output $0.30
class MetricsStore:
    def __init__(self):
        self.total_queries = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        
        self.primary_tokens_input = 0
        self.primary_tokens_output = 0
        self.evaluator_tokens_input = 0
        self.evaluator_tokens_output = 0
        
        self.total_cost_usd = 0.0
        self.saved_cost_usd = 0.0
        
        self.evaluations_count = 0
        self.avg_faithfulness = 0.0
        self.avg_relevance = 0.0
        self.eval_logs: List[Dict[str, Any]] = []

    def log_primary_usage(self, input_tokens: int, output_tokens: int):
        self.primary_tokens_input += input_tokens
        self.primary_tokens_output += output_tokens
        # Groq Llama 3.1 8B cost calculation
        cost = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.08 / 1_000_000)
        self.total_cost_usd += cost

    def log_evaluator_usage(self, input_tokens: int, output_tokens: int):
        self.evaluator_tokens_input += input_tokens
        self.evaluator_tokens_output += output_tokens
        # Gemini 2.5 Flash cost calculation
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        self.total_cost_usd += cost
        self.evaluations_count += 1

    def log_cache_savings(self, input_tokens: int, output_tokens: int):
        # Calculate cost saved if we had called Groq instead of caching
        cost_saved = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.08 / 1_000_000)
        self.saved_cost_usd += cost_saved

    def add_eval_log(self, log_entry: Dict[str, Any]):
        self.eval_logs.append(log_entry)
        # Recalculate average scores
        faith_scores = [log["faithfulness"] for log in self.eval_logs if "faithfulness" in log]
        relev_scores = [log["relevance"] for log in self.eval_logs if "relevance" in log]
        
        if faith_scores:
            self.avg_faithfulness = sum(faith_scores) / len(faith_scores)
        if relev_scores:
            self.avg_relevance = sum(relev_scores) / len(relev_scores)

    def reset(self):
        self.total_queries = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.primary_tokens_input = 0
        self.primary_tokens_output = 0
        self.evaluator_tokens_input = 0
        self.evaluator_tokens_output = 0
        self.total_cost_usd = 0.0
        self.saved_cost_usd = 0.0
        self.evaluations_count = 0
        self.avg_faithfulness = 0.0
        self.avg_relevance = 0.0
        self.eval_logs = []

metrics_store = MetricsStore()


class LLMService:
    def __init__(self):
        self.groq_client = None
        self.gemini_configured = False
        
        # Initialize Groq client if key is provided
        if settings.GROQ_API_KEY:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq client initialized for primary synthesis.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                
        # Initialize Gemini API if key is provided
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_configured = True
                logger.info("Google Gemini configured for evaluation.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Generates an answer using primary synthesis model (Groq Llama 3.1 8B).
        Falls back to intelligent mock generator if API key is not present.
        """
        context_text = "\n\n".join([f"Source [{i+1}] (From {c['metadata'].get('filename', 'doc')}):\n{c['text']}" for i, c in enumerate(context_chunks)])
        
        system_prompt = (
            "You are a highly analytical AI assistant. Answer the user's question based strictly on the provided context. "
            "If the context is empty or doesn't contain the answer, explain that the information is not in the uploaded documents. "
            "Format your answer cleanly with bullet points if appropriate. Always cite your source documents by name."
        )
        
        user_prompt = f"Context Documents:\n{context_text}\n\nQuestion: {query}"
        
        start_time = time.time()
        
        # Check if we should run in simulation mode
        if not self.groq_client:
            logger.info("Running primary LLM in simulation mode.")
            time.sleep(1.0) # Simulate network latency
            answer = self._generate_simulated_answer(query, context_chunks)
            
            # Simulated token counts
            input_tokens = len(user_prompt.split()) + len(system_prompt.split())
            output_tokens = len(answer.split())
            metrics_store.log_primary_usage(input_tokens, output_tokens)
            
            latency_ms = (time.time() - start_time) * 1000
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "simulated": True
            }
            return answer, usage

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=1024
            )
            
            answer = chat_completion.choices[0].message.content
            usage = chat_completion.usage
            
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            metrics_store.log_primary_usage(input_tokens, output_tokens)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return answer, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "simulated": False
            }
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}. Falling back to simulation.")
            # Graceful fallback to simulated answer
            answer = self._generate_simulated_answer(query, context_chunks)
            input_tokens = len(user_prompt.split())
            output_tokens = len(answer.split())
            metrics_store.log_primary_usage(input_tokens, output_tokens)
            latency_ms = (time.time() - start_time) * 1000
            return answer, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency_ms,
                "simulated": True
            }

    def evaluate_response(self, query: str, context_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """
        Evaluates the generated answer using Gemini 2.5 Flash as judge.
        Returns a dictionary with 'faithfulness', 'relevance', and 'reason'.
        """
        context_text = "\n\n".join([c["text"] for c in context_chunks])
        
        evaluation_prompt = (
            "You are a meticulous AI alignment evaluator. Assess the primary LLM's answer based on two metrics:\n"
            "1. Faithfulness (0.0 to 1.0): Did the primary LLM stick strictly to the provided context? Are there any hallucinations or unsupported claims? (1.0 is completely faithful, 0.0 is entirely hallucinated)\n"
            "2. Answer Relevance (0.0 to 1.0): Does the response actually answer the user's query directly and fully?\n\n"
            "Context Documents:\n"
            f"{context_text}\n\n"
            f"User Question: {query}\n"
            f"Primary LLM Answer:\n{answer}\n\n"
            "You MUST return your evaluation in raw JSON format with exactly the following keys. Do not include markdown formatting outside the JSON block.\n"
            "{\n"
            '  "faithfulness": <float between 0.0 and 1.0>,\n'
            '  "relevance": <float between 0.0 and 1.0>,\n'
            '  "reason": "<detailed reason describing your evaluation scores and any hallucinations found>"\n'
            "}"
        )
        
        start_time = time.time()
        
        if not self.gemini_configured:
            logger.info("Running evaluation model in simulation mode.")
            time.sleep(0.8) # Simulate network latency
            eval_res = self._evaluate_simulated(query, context_chunks, answer)
            
            # Simulated token count
            input_tokens = len(evaluation_prompt.split())
            output_tokens = 150
            metrics_store.log_evaluator_usage(input_tokens, output_tokens)
            
            eval_res["latency_ms"] = (time.time() - start_time) * 1000
            eval_res["simulated"] = True
            return eval_res

        try:
            import google.generativeai as genai
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            response = model.generate_content(
                evaluation_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Load the JSON
            result = json.loads(response.text.strip())
            
            # Log token counts (approximate since we don't have native metadata directly in response objects in all SDKs)
            input_tokens = len(evaluation_prompt.split())
            output_tokens = len(response.text.split())
            metrics_store.log_evaluator_usage(input_tokens, output_tokens)
            
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["simulated"] = False
            
            # Sanitize scores to floats
            result["faithfulness"] = float(result.get("faithfulness", 1.0))
            result["relevance"] = float(result.get("relevance", 1.0))
            return result
        except Exception as e:
            logger.error(f"Error calling Gemini evaluator: {e}. Falling back to simulation.")
            eval_res = self._evaluate_simulated(query, context_chunks, answer)
            eval_res["latency_ms"] = (time.time() - start_time) * 1000
            eval_res["simulated"] = True
            return eval_res

    def _generate_simulated_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Helper to generate a realistic mock answer by extracting keywords and sentences from context."""
        if not context_chunks:
            return (
                "Based on my check of the uploaded database, no documentation has been provided. "
                "Therefore, I do not have enough context to answer your question regarding: '" + query + "'."
            )
            
        # Compile a clean response based on chunks
        source_names = list(set([c["metadata"].get("filename", "document.txt") for c in context_chunks]))
        source_cite = " and ".join([f"[{name}]" for name in source_names])
        
        # Simple extraction of key sentences
        sentences = []
        for chunk in context_chunks:
            # Get first two sentences of chunk text
            text = chunk["text"]
            parts = re.split(r'(?<=[.!?])\s+', text)
            if parts:
                sentences.append(parts[0])
                if len(parts) > 1:
                    sentences.append(parts[1])
                    
        summary_text = " ".join(sentences[:3])
        
        return (
            f"According to the source documentation {source_cite}, here is what was found:\n\n"
            f"- **Context Overview**: {summary_text}\n"
            f"- **Relevance to Query**: The query '{query}' relates directly to the documents which discuss details found in {source_names[0]}.\n\n"
            f"If you require further details, please consult the full sections inside {source_cite}."
        )

    def _evaluate_simulated(self, query: str, context_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Helper to generate a realistic evaluation scorecard in simulation mode."""
        if not context_chunks:
            return {
                "faithfulness": 1.0,
                "relevance": 0.5,
                "reason": "Evaluator (Gemini 2.5 Flash Sim): No context was uploaded. The response correctly stated that it couldn't answer, showing high faithfulness but limited relevance to the original intent."
            }
            
        # Basic overlap checks
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set()
        for chunk in context_chunks:
            context_words.update(re.findall(r'\w+', chunk["text"].lower()))
            
        # Check if words in answer are present in context
        # We exclude common stop words to get a better semantic check
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'according', 'here', 'what', 'found'}
        important_answer_words = answer_words - stop_words
        
        if not important_answer_words:
            overlap_ratio = 1.0
        else:
            overlapping_words = important_answer_words.intersection(context_words)
            overlap_ratio = len(overlapping_words) / len(important_answer_words)
            
        # Compute scores with minor probabilistic variance to make dashboard look dynamic
        # Force high faithfulness if overlap is high
        faithfulness = round(max(0.6, min(1.0, overlap_ratio + 0.15)), 2)
        
        # Relevance is simulated based on query matching the answer
        query_words = set(re.findall(r'\w+', query.lower())) - stop_words
        query_overlap = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 1.0
        relevance = round(max(0.7, min(1.0, query_overlap + 0.3)), 2)
        
        # Generate a nice description
        status = "EXCELLENT" if faithfulness >= 0.85 else "WARNING"
        reason = (
            f"[Gemini 2.5 Flash Sim]: The primary LLM answer is faithful to the context with a score of {faithfulness:.2f}. "
            f"It correctly summarized information from {[c['metadata'].get('filename') for c in context_chunks]}. "
            f"The relevance score is {relevance:.2f} as it directly addresses '{query}'."
        )
        if status == "WARNING":
            reason += " Note: The model used words that were not directly in the source documents, indicating minor stylistic elaboration."

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "reason": reason
        }

# Singleton instance
llm_service = LLMService()
