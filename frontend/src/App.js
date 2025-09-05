import React, { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!query) return;

    setLoading(true);
    setAnswer("");

    try {
      const res = await axios.post("http://localhost:8000/ask", { question: query });
      setAnswer(res.data.answer);
    } catch (error) {
      console.error(error);
      setAnswer("⚠️ Error fetching answer from backend.");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2>🧠 Mini RAG Assistant</h2>

      <div style={{ marginBottom: "10px" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          style={{ width: "400px", padding: "8px", marginRight: "10px" }}
        />
        <button onClick={handleAsk} style={{ padding: "8px 12px" }}>
          Ask
        </button>
      </div>

      {loading && <p>⏳ Generating answer...</p>}

      {answer && (
        <div
          style={{
            border: "1px solid #ddd",
            padding: "10px",
            borderRadius: "5px",
            maxHeight: "400px",
            overflowY: "auto",
            whiteSpace: "pre-wrap",
            backgroundColor: "#f9f9f9",
          }}
        >
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
