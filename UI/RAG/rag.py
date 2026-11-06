"""rag.py — all RAG + LLM logic. No FastAPI, no Streamlit."""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
import pymupdf
from docx import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
TEXT_DIR = BASE_DIR / "data" / "extracted_text"
VECTOR_DB_ROOT = BASE_DIR / "vector_db"
for _d in (UPLOAD_DIR, TEXT_DIR, VECTOR_DB_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("api_key") or os.getenv("GROQ_API_KEY")
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

# ---------------------------------------------------------------------------
# IN-MEMORY SESSION STATE (per-process; holds docs + current test)
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


def _sess(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {"documents": [], "current_test": None}
    return _sessions[sid]


# ---------------------------------------------------------------------------
# EMBEDDINGS (singleton)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------------
def _extract(path: Path, text_path: Path) -> int:
    ext = path.suffix.lower()
    text_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".pdf":
        doc = pymupdf.open(path)
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                for n, page in enumerate(doc):
                    t = page.get_text("text").strip()
                    if t:
                        f.write(f"\n\n PAGE {n + 1} \n\n{t}")
            return len(doc)
        finally:
            doc.close()

    if ext == ".docx":
        d = Document(str(path))
        with open(text_path, "w", encoding="utf-8") as f:
            for p in d.paragraphs:
                if p.text.strip():
                    f.write(p.text + "\n")
        return 1

    if ext == ".txt":
        text_path.write_text(
            path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
        )
        return 1

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------
def _vs_dir(sid: str) -> Path:
    d = VECTOR_DB_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_vs(sid: str) -> Optional[Chroma]:
    d = _vs_dir(sid)
    if not any(d.iterdir()):
        return None
    return Chroma(persist_directory=str(d), embedding_function=_embeddings())


# ---------------------------------------------------------------------------
# GROQ LLM
# ---------------------------------------------------------------------------
def _call_llm(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: float = 90.0,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ API key missing (env: api_key or GROQ_API_KEY).")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = httpx.post(LLM_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def upload_and_index(sid: str, filename: str, file_bytes: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED_EXTS)}")

    upload_path = UPLOAD_DIR / f"{sid}__{filename}"
    upload_path.write_bytes(file_bytes)

    text_path = TEXT_DIR / f"{sid}__{Path(filename).stem}.txt"
    pages = _extract(upload_path, text_path)

    if not text_path.exists() or text_path.stat().st_size == 0:
        raise ValueError("No extractable text found in this file.")

    docs = TextLoader(str(text_path), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=str(_vs_dir(sid)),
    )

    s = _sess(sid)
    if filename not in s["documents"]:
        s["documents"].append(filename)

    return {"filename": filename, "pages": pages, "chunks": len(chunks)}


def ask(sid: str, question: str) -> str:
    vs = _load_vs(sid)
    if vs is None:
        raise ValueError("No knowledge base for this session. Upload a document first.")

    hits = vs.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in hits) or "(none)"

    prompt = f"""You are an AI Study Assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

CONTEXT:
{context}

QUESTION: {question}
"""
    return _call_llm(prompt, temperature=0.2, max_tokens=600)


def generate_test(sid: str, num_questions: int) -> list[dict]:
    vs = _load_vs(sid)
    if vs is None:
        raise ValueError("No knowledge base for this session. Upload a document first.")

    hits = vs.similarity_search(
        "summary of all main concepts, definitions, and important facts", k=10
    )
    context = "\n\n".join(d.page_content for d in hits) or "(none)"

    prompt = f"""You are an expert examiner. Based on the context below, generate a
multiple-choice test with exactly {num_questions} questions.

Return ONLY valid JSON (no markdown code fences) in this exact shape:
[
  {{"question": "The question text?", "options": ["A", "B", "C"], "correct_index": 0}}
]

CONTEXT:
{context}
"""
    raw = _call_llm(prompt, temperature=0.3, max_tokens=4000, timeout=120)
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI returned invalid JSON: {e}")

    questions = []
    for item in data:
        q = item.get("question")
        opts = item.get("options") or []
        ci = item.get("correct_index", 0)
        if not q or len(opts) < 2 or not (0 <= ci < len(opts)):
            continue
        questions.append(
            {"question": q, "options": opts, "correct_index": int(ci)}
        )

    if len(questions) < max(1, num_questions // 2):
        raise ValueError("AI produced too few valid questions. Please try again.")

    _sess(sid)["current_test"] = questions

    # Return to client WITHOUT correct answers
    return [{"question": q["question"], "options": q["options"]} for q in questions]


def submit_test(sid: str, answers: list[Optional[int]]) -> dict:
    s = _sess(sid)
    questions = s.get("current_test")
    if not questions:
        raise ValueError("No active test for this session.")
    if len(answers) != len(questions):
        raise ValueError(f"Expected {len(questions)} answers, got {len(answers)}.")

    score = 0
    results = []
    for i, q in enumerate(questions):
        u = answers[i]
        c = q["correct_index"]
        ok = u is not None and u == c
        if ok:
            score += 1
        results.append(
            {
                "question": q["question"],
                "options": q["options"],
                "correct_index": c,
                "user_index": u,
                "is_correct": ok,
            }
        )

    total = len(questions)
    pct = round(score / total * 100, 1) if total else 0.0
    s["current_test"] = None

    return {
        "score": score,
        "total": total,
        "percentage": pct,
        "passed": pct >= 60,
        "results": results,
    }


def status(sid: str) -> dict:
    s = _sess(sid)
    vs = _load_vs(sid)
    return {
        "has_kb": vs is not None and bool(s["documents"]),
        "documents": list(s["documents"]),
    }


def clear(sid: str) -> None:
    d = VECTOR_DB_ROOT / sid
    if d.exists():
        shutil.rmtree(d)
    _sessions.pop(sid, None)