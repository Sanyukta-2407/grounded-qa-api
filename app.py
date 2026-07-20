import json
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOT_ANSWERABLE = "I don't know"

_client = None


def get_client():
    """Lazily construct the OpenAI client so a missing API key doesn't
    crash the whole app at import/startup time."""
    global _client
    if _client is None:
        api_key = os.getenv("AIPIPE_API_KEY")
        if not api_key:
            raise RuntimeError("AIPIPE_API_KEY environment variable is not set")
        _client = OpenAI(api_key=api_key, base_url="https://aipipe.org/openai/v1")
    return _client


class Chunk(BaseModel):
    chunk_id: str
    text: str


class QARequest(BaseModel):
    question: str
    chunks: List[Chunk] = []


def not_answerable_response(confidence: float = 0.0):
    return {
        "answer": NOT_ANSWERABLE,
        "citations": [],
        "confidence": max(0.0, min(confidence, 0.3)),
        "answerable": False,
    }


@app.get("/")
def home():
    return {"status": "Grounded QA API is running"}


@app.post("/")
def grounded_qa(request: QARequest):
    # Guard: empty/malformed input
    if not request.question or not request.question.strip() or not request.chunks:
        return not_answerable_response()

    context = "\n\n".join(
        f"Chunk ID: {chunk.chunk_id}\n{chunk.text}"
        for chunk in request.chunks
    )

    prompt = f"""
You are a grounded question answering system.

Answer ONLY using the information contained in the chunks below.

Rules:
1. Never use outside knowledge.
2. If the answer is not completely supported by the chunks, set "answerable" to false,
   set "answer" to "I don't know", and set "citations" to an empty list.
3. Cite ONLY chunk IDs that directly support the answer.
4. Do NOT cite irrelevant chunks.
5. If multiple chunks support the answer, include all of them.
6. Return ONLY valid JSON, with no markdown fences and no extra text.

Question:
{request.question}

Chunks:
{context}

Return exactly this JSON shape:

{{
  "answer": "...",
  "citations": ["chunk_id"],
  "confidence": 0.0,
  "answerable": true
}}
"""

    try:
        response = get_client().chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded QA assistant. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=10,
        )

        output = response.choices[0].message.content.strip()

        # Strip markdown fences if the model added them anyway
        if output.startswith("```"):
            lines = output.splitlines()
            lines = [line for line in lines if not line.startswith("```")]
            output = "\n".join(lines)

        result = json.loads(output)

        answer = str(result.get("answer", "")).strip()
        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        valid_ids = {chunk.chunk_id for chunk in request.chunks}
        citations = [
            cid for cid in result.get("citations", []) if cid in valid_ids
        ]

        # Strict unanswerable rule: no citations or model says it doesn't know
        if not answer or answer.strip().lower() == NOT_ANSWERABLE.lower() or not citations:
            return not_answerable_response(confidence)

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "answerable": True,
        }

    except Exception:
        return not_answerable_response()