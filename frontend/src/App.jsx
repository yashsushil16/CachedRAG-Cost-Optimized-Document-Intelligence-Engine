import React, { useState, useEffect, useRef } from "react";
import { 
  Send, 
  UploadCloud, 
  Database, 
  Cpu, 
  Zap, 
  Clock, 
  ShieldAlert, 
  RefreshCw, 
  FileText,
  TrendingUp,
  DollarSign
} from "lucide-react";
import "./App.css";

const API_BASE = "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! Upload your text or PDF documentation in the left sidebar, and ask me questions. I will leverage a local semantic cache (Redis-style VSS) to answer instantly if you ask something similar, otherwise I will query the Qdrant vector database and synthesize a response via Groq. The results are audited in real-time by a Gemini 2.5 Flash judge.",
      source: "system",
      latency_ms: 0,
      cache_similarity: 0,
      retrieved_chunks: []
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [metrics, setMetrics] = useState({
    summary: { total_queries: 0, cache_hits: 0, cache_misses: 0, cache_hit_ratio: 0.0, cached_items: 0 },
    llm_usage: { primary_input_tokens: 0, primary_output_tokens: 0, evaluator_input_tokens: 0, evaluator_output_tokens: 0, total_cost_usd: 0.0, saved_cost_usd: 0.0 },
    evaluation_metrics: { evaluations_count: 0, average_faithfulness: 0.0, average_relevance: 0.0, logs: [] },
    system_status: { simulation_mode: true, groq_active: false, gemini_active: false }
  });

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Initial load & Polling for stats
  useEffect(() => {
    fetchDocuments();
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Error fetching metrics:", err);
    }
  };

  // Upload handler
  const handleFileUpload = async (file) => {
    if (!file) return;
    if (!file.name.endsWith(".pdf") && !file.name.endsWith(".txt")) {
      alert("Only PDF and TXT files are supported.");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        fetchDocuments();
        fetchMetrics();
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            content: `Successfully ingested and parsed "${file.name}". Created and indexed ${data.chunks_count} vector chunks in the local Qdrant memory database.`,
            source: "system",
            latency_ms: 0,
            cache_similarity: 0,
            retrieved_chunks: []
          }
        ]);
      } else {
        alert(`Upload failed: ${data.detail}`);
      }
    } catch (err) {
      console.error("Error uploading file:", err);
      alert("Network error uploading document.");
    } finally {
      setUploading(false);
    }
  };

  // Drag & Drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelectChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  // Send Message
  const handleSendMessage = async (e) => {
    e.preventDefault();
    const text = query.trim();
    if (!text || loading) return;

    setQuery("");
    // Add user message immediately
    setMessages(prev => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });

      const data = await response.json();
      if (response.ok) {
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            source: data.source,
            latency_ms: data.latency_ms,
            cache_similarity: data.cache_similarity,
            retrieved_chunks: data.retrieved_chunks,
            evaluation: data.evaluation
          }
        ]);
        fetchMetrics();
      } else {
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${data.detail || "Could not process request."}`,
            source: "system",
            latency_ms: 0,
            cache_similarity: 0,
            retrieved_chunks: []
          }
        ]);
      }
    } catch (err) {
      console.error("Error sending query:", err);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "Network error. Failed to reach the RAG gateway.",
          source: "system",
          latency_ms: 0,
          cache_similarity: 0,
          retrieved_chunks: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Reset metrics & cache
  const handleReset = async () => {
    if (!window.confirm("Are you sure you want to clear all index documents, semantic caches, and reset usage metrics?")) return;
    try {
      const response = await fetch(`${API_BASE}/api/metrics/reset`, { method: "POST" });
      if (response.ok) {
        setMessages([
          {
            role: "assistant",
            content: "All data stores have been reset. Databases, cache, and usage stats are cleared.",
            source: "system",
            latency_ms: 0,
            cache_similarity: 0,
            retrieved_chunks: []
          }
        ]);
        setDocuments([]);
        fetchMetrics();
      }
    } catch (err) {
      console.error("Error resetting data:", err);
    }
  };

  // Helper for rendering SVG Circular Gauges
  const renderGauge = (score, label, type) => {
    const radius = 28;
    const circ = 2 * Math.PI * radius;
    const offset = circ - (score * circ);
    const colorClass = type === "faith" ? "faithfulness" : "relevance";

    return (
      <div className="gauge-item">
        <div className="gauge-svg-container">
          <svg className="gauge-circle" width="70" height="70">
            <circle className="gauge-bg" cx="35" cy="35" r={radius} />
            <circle 
              className={`gauge-fill ${colorClass}`} 
              cx="35" 
              cy="35" 
              r={radius} 
              strokeDasharray={circ}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="gauge-text">{Math.round(score * 100)}%</div>
        </div>
        <span className="gauge-label">{label}</span>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="logo-section">
          <Database size={24} className="text-violet-500" />
          <h1>AURA Architecture</h1>
          <div className={`system-badge ${metrics.system_status.simulation_mode ? "simulation" : ""}`}>
            <span className={`w-2 h-2 rounded-full ${metrics.system_status.simulation_mode ? "bg-amber-400" : "bg-emerald-400"}`}></span>
            {metrics.system_status.simulation_mode ? "SIMULATION MODE" : "PRODUCTION MODE"}
          </div>
        </div>
        <button onClick={handleReset} className="btn-reset" title="Wipe Vector DB & Cache">
          <RefreshCw size={14} /> Reset System
        </button>
      </header>

      {/* Main Layout Grid */}
      <div className="app-grid">
        
        {/* Left Panel - Ingestion dropzone & doc tracker */}
        <div className="panel-card">
          <div className="panel-header">
            <UploadCloud size={18} className="upload-icon" />
            <h2>Knowledge Base</h2>
          </div>
          <div className="panel-body">
            <div 
              className={`upload-dropzone ${dragActive ? "dragging" : ""}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileSelectChange} 
                style={{ display: "none" }}
                accept=".txt,.pdf"
              />
              {uploading ? (
                <>
                  <div className="spinner"></div>
                  <span>Chunking & Indexing...</span>
                </>
              ) : (
                <>
                  <UploadCloud size={32} />
                  <p>Drop PDF / TXT files</p>
                  <span>Or click to browse</span>
                </>
              )}
            </div>

            <div className="doc-list-title">Indexed Files ({documents.length})</div>
            <div className="flex-1 overflow-y-auto flex flex-col gap-2" style={{ maxHeight: "240px" }}>
              {documents.length === 0 ? (
                <div className="no-docs-message">No files ingested.</div>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} className="document-item">
                    <div className="doc-info">
                      <FileText size={14} className="text-cyan-400" />
                      <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                    </div>
                    <span className="doc-badge">Qdrant</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Middle Panel - Live Chat interface */}
        <div className="panel-card chat-window">
          <div className="panel-header">
            <Cpu size={18} className="text-violet-500" />
            <h2>Interactive Gateway Chat</h2>
          </div>
          
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message-bubble ${msg.role}`}>
                {msg.content}
                
                {/* Metadatas for assistant replies */}
                {msg.role === "assistant" && msg.source !== "system" && (
                  <div className="message-meta">
                    <span className={`source-tag ${msg.source}`}>
                      {msg.source === "cache" ? <Zap size={10} /> : <Cpu size={10} />}
                      {msg.source === "cache" ? "Semantic Cache Hit" : "RAG Synthesis"}
                    </span>
                    
                    <span className="flex items-center gap-1">
                      <Clock size={10} /> {msg.latency_ms.toFixed(1)} ms
                    </span>

                    {msg.source === "cache" && (
                      <span>Match Similarity: {(msg.cache_similarity * 100).toFixed(1)}%</span>
                    )}
                  </div>
                )}

                {/* Display retrieved sources if context was checked */}
                {msg.role === "assistant" && msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                  <div className="chunks-reference">
                    <div className="chunks-ref-title">Retrieved Chunks from Qdrant:</div>
                    {msg.retrieved_chunks.map((chunk, cIdx) => (
                      <div key={cIdx} className="chunk-card">
                        <strong>[{chunk.filename} (Score: {chunk.score.toFixed(3)})]</strong>: {chunk.text.substring(0, 160)}...
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="message-bubble assistant">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <form onSubmit={handleSendMessage} className="chat-form">
              <input
                type="text"
                className="chat-input"
                placeholder="Ask something about uploaded documents..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={loading}
              />
              <button type="submit" className="btn-send" disabled={!query.trim() || loading}>
                <Send size={16} />
              </button>
            </form>
          </div>
        </div>

        {/* Right Panel - Real-time metrics & evaluation dashboard */}
        <div className="panel-card">
          <div className="panel-header">
            <TrendingUp size={18} className="text-cyan-400" />
            <h2>Live Architecture Analytics</h2>
          </div>
          <div className="panel-body">
            
            {/* Cache Stats */}
            <div className="stats-grid">
              <div className="stat-box">
                <div className="stat-label">Cache Hit Ratio</div>
                <div className="stat-value green">
                  {(metrics.summary.cache_hit_ratio * 100).toFixed(0)}%
                </div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Total Queries</div>
                <div className="stat-value purple">{metrics.summary.total_queries}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Cache Hits</div>
                <div className="stat-value green">{metrics.summary.cache_hits}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Cache Misses</div>
                <div className="stat-value yellow">{metrics.summary.cache_misses}</div>
              </div>
            </div>

            {/* Cost Tracker */}
            <div className="stat-box" style={{ width: '100%', textAlign: 'left', padding: '12px 16px' }}>
              <div className="stat-label flex justify-between items-center">
                <span>API Computation Cost</span>
                <span className="flex items-center text-emerald-400 gap-1 font-semibold text-xs">
                  <TrendingUp size={12} /> Saved: ${metrics.llm_usage.saved_cost_usd.toFixed(5)}
                </span>
              </div>
              <div className="flex items-baseline gap-1 mt-2">
                <DollarSign size={18} className="text-cyan-400" />
                <span className="font-display font-bold text-xl">{metrics.llm_usage.total_cost_usd.toFixed(6)}</span>
                <span className="text-xs text-gray-500">USD</span>
              </div>
            </div>

            {/* Latency Comparison Indicator */}
            <div className="latency-container">
              <div className="latency-bar-label">
                <span>Semantic Cache Hit latency</span>
                <strong>&lt; 10 ms</strong>
              </div>
              <div className="latency-track">
                <div className="latency-fill cache" style={{ width: "8%" }}></div>
              </div>

              <div className="latency-bar-label mt-2">
                <span>Groq Llama 3.1 Inference latency</span>
                <strong>~ 1,200 ms</strong>
              </div>
              <div className="latency-track">
                <div className="latency-fill llm" style={{ width: "90%" }}></div>
              </div>
            </div>

            {/* Alignment Judge Circular Gauges */}
            <div className="panel-header" style={{ padding: '8px 0', borderBottom: 'none' }}>
              <Cpu size={14} className="text-purple-400" />
              <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f3f4f6' }}>Gemini Alignment Judge Scorecard</h3>
            </div>
            
            <div className="gauges-row">
              {renderGauge(metrics.evaluation_metrics.average_faithfulness, "Faithfulness", "faith")}
              {renderGauge(metrics.evaluation_metrics.average_relevance, "Relevance", "relevance")}
            </div>

            {/* Real-time Judge logs */}
            <div className="panel-header" style={{ padding: '8px 0', borderBottom: 'none', marginTop: '4px' }}>
              <Clock size={14} className="text-cyan-400" />
              <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f3f4f6' }}>Audit Trail (Evaluation Logs)</h3>
            </div>

            <div className="eval-logs-container">
              {metrics.evaluation_metrics.logs.length === 0 ? (
                <div className="no-docs-message" style={{ padding: '8px 0' }}>No evaluation audits logged yet.</div>
              ) : (
                metrics.evaluation_metrics.logs.map((log, lIdx) => (
                  <div key={lIdx} className="eval-log-card">
                    <div className="eval-log-header">
                      <span className="eval-log-query" title={log.query}>Q: {log.query}</span>
                      <div className="eval-scores">
                        <span className="eval-score-badge faith">F: {log.faithfulness.toFixed(1)}</span>
                        <span className="eval-score-badge relev">R: {log.relevance.toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="eval-log-reason">{log.reason}</div>
                  </div>
                ))
              )}
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
