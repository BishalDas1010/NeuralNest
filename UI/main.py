"""main.py — FastAPI backend."""

import shutil
import uuid
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import RAG.rag as rag 

app = FastAPI(title="StudyRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    session_id: str
    question: str

class GenerateReq(BaseModel):
    session_id: str
    num_questions: int = Field(..., ge=5, le=30)

class SubmitReq(BaseModel):
    session_id: str
    answers: list[Optional[int]]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
def upload(session_id: str = Query(...), file: UploadFile = File(...)):
    try:
        file_extension = Path(file.filename).suffix
        temp_file_path = rag.UPLOAD_DIR / f"{uuid.uuid4()}{file_extension}"
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Returns dict with filename, pages, chunks
        result = rag.upload_document(temp_file_path, session_id=session_id)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

@app.get("/api/session/{sid}/status")
def session_status(sid: str):
    return rag.status(sid)

@app.delete("/api/session/{sid}")
def session_clear(sid: str):
    rag.clear_session(sid)
    return {"ok": True}

@app.post("/api/chat")
def chat(req: ChatReq):
    try:
        answer = rag.ask_question(question=req.question, session_id=req.session_id)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")

@app.post("/api/mock-test/generate")
def generate(req: GenerateReq):
    try:
        return {"questions": rag.generate_test(req.session_id, req.num_questions)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

@app.post("/api/mock-test/submit")
def submit(req: SubmitReq):
    try:
        return rag.submit_test(req.session_id, req.answers)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Submit failed: {e}")