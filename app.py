from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

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
    try:
        if not request.question.strip() or len(request.chunks) == 0:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.0,
                "answerable": False
            }

        stopwords = {
            "what", "when", "where", "who", "which", "why", "how",
            "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an", "of", "to", "in", "on", "for",
            "and", "or", "did", "does", "do", "at", "by",
            "with", "from", "into", "about", "than", "then",
            "this", "that", "these", "those", "it", "its",
            "year"
        }

        keywords = []

        for word in request.question.lower().split():
            word = word.strip(".,?!:;()[]{}\"'")
            if word and word not in stopwords:
                keywords.append(word)

        best_chunk = None
        best_score = 0

        for chunk in request.chunks:
            text = chunk.text.lower()
            score = 0

            for word in keywords:
                if word in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is None or best_score == 0:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.2,
                "answerable": False
            }

        confidence = 0.5 + 0.1 * best_score
        if confidence > 0.95:
            confidence = 0.95

        return {
            "answer": best_chunk.text,
            "citations": [best_chunk.chunk_id],
            "confidence": round(confidence, 2),
            "answerable": True
        }

    except Exception:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }