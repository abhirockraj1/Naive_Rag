# 📚 Naive RAG — Handbook Q&A Assistant

A simple Retrieval-Augmented Generation (RAG) pipeline that answers questions from a company handbook PDF using **SmolLM2** running locally via **llama.cpp**.

---

## 🏗️ Architecture

```
handbook_qa_brief.pdf
        │
        ▼
┌──────────────────┐
│  1. PDF Loading   │  PyMuPDF extracts text from PDF
└────────┬─────────┘
         ▼
┌──────────────────┐
│  2. Chunking      │  Split text into overlapping 300-char chunks
└────────┬─────────┘
         ▼
┌──────────────────┐
│  3. Embedding     │  SmolLM2-135M encodes chunks into 576-dim vectors
└────────┬─────────┘
         ▼
┌──────────────────┐
│  4. Vector Search │  Cosine similarity finds top-k relevant chunks
└────────┬─────────┘
         ▼
┌──────────────────┐
│  5. LLM Answer    │  SmolLM2 via llama.cpp generates cited answer
└──────────────────┘
```

---

## 🛠️ Prerequisites

- **Python 3.13+**
- **llama.cpp** with SmolLM2 GGUF model
- A virtual environment (`.venv`)

### Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install pymupdf
pip install sentence-transformers
pip install openai
```

### Start llama.cpp server

```powershell
llama-server.exe -m smollm2-135m.gguf -c 16384 --port 9931
```

> Use `-c 16384` to set a 16K context window. Default 8192 may be too small when sending multiple chunks.

---

## 📖 Pipeline Steps

### Step 1 — Load PDF

Extract raw text from the handbook PDF using PyMuPDF.

```python
import pymupdf

doc = pymupdf.open("handbook_qa_brief.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()
```

---

### Step 2 — Chunking

Split the extracted text into overlapping chunks of 300 characters with a 50-character overlap. This ensures no information is lost at chunk boundaries.

```python
def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "id": len(chunks),
            "text": chunk.strip(),
            "start": start,
            "end": min(end, len(text))
        })
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text, chunk_size=300, overlap=50)
```

**Result:** 7 chunks from a single-page PDF.

| Chunk | Content Preview |
|-------|----------------|
| 0 | Handbook Q&A Assistant – Project brief... |
| 1 | ...ask questions in plain English and get instant, cited answers... |
| 2 | ..."What is the notice period?" Before and after... |
| 3 | ...Seconds per lookup, Output, A short direct answer... |
| 4 | ...Suggested approach, Ingest: extract text from the PDF... |
| 5 | ...Answer: have an LLM respond using only the retrieved passages... |
| 6 | ...citation the employee can verify, Out-of-scope questions... |

---

### Step 3 — Embedding

Encode each chunk into a 576-dimensional vector using **SmolLM2-135M** via the `sentence-transformers` library.

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("HuggingFaceTB/SmolLM2-135M")
embed_model.tokenizer.pad_token = embed_model.tokenizer.eos_token

chunk_texts = [c["text"] for c in chunks]
embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)
```

> **Note:** SmolLM2 is a causal language model, not a dedicated embedding model. Its tokenizer lacks a padding token by default, so we set `pad_token = eos_token`.

**Output:** `embeddings.shape = (7, 576)`

---

### Step 4 — Vector Search

Given a user query, encode it with the same model and find the most similar chunks using **cosine similarity**.

```python
import numpy as np

def vector_search(query, embed_model, embeddings, chunks, top_k=2):
    """Search for the most relevant chunks given a query."""
    query_embedding = embed_model.encode([query])

    scores = np.dot(embeddings, query_embedding.T).squeeze()
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = scores / norms

    top_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunks[idx]["id"],
            "score": float(cosine_scores[idx]),
            "text": chunks[idx]["text"]
        })
    return results
```

**Example results:**

| Query | Top Chunk | Score |
|-------|-----------|-------|
| "How many sick leaves do I get?" | Chunk 5 | 0.8361 |
| "What is the notice period?" | Chunk 6 | 0.8933 |
| "What is the suggested approach?" | Chunk 6 | 0.8980 |

---

### Step 5 — LLM Inference (RAG Answer)

Send the retrieved chunks as context along with the user's question to **SmolLM2** running locally via **llama.cpp**'s OpenAI-compatible API.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9931/v1",
    api_key="sk-no-key-required"
)

def rag_answer(query, embed_model, embeddings, chunks, top_k=3):
    """Retrieve relevant chunks and generate an answer using SmolLM2."""

    # Retrieve
    results = vector_search(query, embed_model, embeddings, chunks, top_k=top_k)

    # Build context
    context = "\n\n".join(
        [f"[Chunk {r['chunk_id']}]: {r['text']}" for r in results]
    )

    # Generate
    system_prompt = (
        "You are a helpful handbook Q&A assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Include a citation referencing the chunk number. "
        "If the context doesn't contain the answer, say "
        "'The handbook does not cover this topic.'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="smollm2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=200,
        temperature=0.3,
        top_p=0.95
    )

    answer = response.choices[0].message.content
    return answer, results
```

**Example output:**

```
🔍 Query: What is the suggested approach for building this assistant?

💬 Answer:
The suggested approach is to use a combination of NLP and ML techniques...

📚 Sources:
  Chunk 6 (score: 0.9051): ...citation the employee can verify...
  Chunk 5 (score: 0.8962): ...Answer: have an LLM respond using only the retrieved passages...
```

---

## 📁 Project Structure

```
Naive_Rag/
├── Naive_RAG.ipynb          # Main notebook with the full pipeline
├── handbook_qa_brief.pdf    # Input PDF (company handbook project brief)
└── README.md                # This file
```

---

## ⚠️ Known Limitations

- **SmolLM2 as embeddings:** SmolLM2 is a causal LM, not a dedicated embedding model. Retrieval quality could improve with purpose-built models like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`.
- **Fixed-size chunking:** Splits at arbitrary character boundaries, potentially cutting mid-sentence. Semantic chunking (splitting by topic/meaning) would produce better results.
- **Context window:** Default llama.cpp context is 8192 tokens. Use `-c 16384` or higher when sending multiple chunks.
- **Small model:** SmolLM2-135M is very lightweight. Larger models (360M, 1.7B) or other LLMs will produce more coherent answers.

---

## 🚀 Next Steps

- [ ] Replace fixed chunking with **semantic chunking**
- [ ] Use a dedicated embedding model (e.g. `all-MiniLM-L6-v2`)
- [ ] Add a **vector database** (FAISS, ChromaDB) for larger documents
- [ ] Test with a real 30-page company handbook
- [ ] Add a simple **Gradio/Streamlit UI**
