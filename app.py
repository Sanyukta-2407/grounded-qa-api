from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]

@app.get("/")
def home():
    return {"status": "Grounded QA API is running"}

@app.post("/")
def grounded_qa(request: QARequest):
    question = request.question.lower()

    # Handle empty input
    if not request.question.strip() or len(request.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    matched = []

    # Very simple keyword matching
    keywords = [
        w.strip(".,?!")
        for w in question.split()
        if len(w) > 2
    ]

    for chunk in request.chunks:
        text = chunk.text.lower()
        if any(k in text for k in keywords):
            matched.append(chunk)

    if not matched:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False
        }

    answer = " ".join(chunk.text for chunk in matched)

    return {
        "answer": answer,
        "citations": [chunk.chunk_id for chunk in matched],
        "confidence": round(min(0.95, 0.6 + 0.1 * len(matched)), 2),
        "answerable": True
    }