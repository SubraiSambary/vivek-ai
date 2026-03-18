# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import shutil, os
from dotenv import load_dotenv

load_dotenv()

from backend.memory       import init_db
from backend.vivek_brain  import chat
from backend.vector_store import ingest_pdf, ingest_text, get_doc_count

app = FastAPI(title="VIVEK API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    os.makedirs("data/uploads", exist_ok=True)
    print("🧠 VIVEK is awake and ready!")

class ChatRequest(BaseModel):
    user_id:   str
    message:   str
    user_name: Optional[str] = None

@app.get("/")
def root():
    return {"message": "VIVEK API is running! Hit /health to check status."}

@app.get("/health")
def health():
    return {"status": "VIVEK is alive and caffeinated ☕"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        result = chat(
            user_id=req.user_id,
            message=req.message,
            user_name=req.user_name,
        )
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"reply": f"Arrey yaar, something broke internally! 😅 Error: {str(e)}",
                     "reaction": {"type": "emoji", "text": "😵‍💫"},
                     "used_web": False, "used_docs": False}
        )

@app.post("/upload")
async def upload_file(
    file:    UploadFile = File(...),
    user_id: str        = Form("default"),
):
    try:
        save_path = f"data/uploads/{user_id}_{file.filename}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.lower().endswith(".pdf"):
            chunks = ingest_pdf(save_path, user_id=user_id)
        else:
            content = open(save_path, "r", encoding="utf-8", errors="ignore").read()
            chunks  = ingest_text(content, source=file.filename, user_id=user_id)

        return {"status": "success",
                "message": f"Ingested {chunks} chunks from {file.filename}",
                "chunks": chunks}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/doc-count/{user_id}")
def doc_count(user_id: str):
    try:
        return {"count": get_doc_count(user_id)}
    except Exception:
        return {"count": 0}