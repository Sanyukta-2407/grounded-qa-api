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

client = OpenAI(
    api_key=os.getenv("AIPIPE_API_KEY"),
    base_url="https://aipipe.org/openai/v1",
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
    if not request.question.strip() or not request.chunks:
        return {
            "answer": "I don't know.",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    context = "\n\n".join(
        f"Chunk ID: {chunk.chunk_id}\n{chunk.text}"
        for chunk in request.chunks
    )

    prompt = f"""
You are a grounded question answering system.

Answer ONLY using the information contained in the chunks below.

Rules:
1. Never use outside knowledge.
2. If the answer is not completely supported by the chunks, answer "I don't know."
3. Cite ONLY chunk IDs that directly support the answer.
4. Do NOT cite irrelevant chunks.
5. If multiple chunks support the answer, include all of them.
6. Return ONLY valid JSON.

Question:
{request.question}

Chunks:
{context}

Return exactly this JSON:

{{
  "answer":"...",
  "citations":["chunk_id"],
  "confidence":0.0,
  "answerable":true
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded QA assistant. Respond only with valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        output = response.choices[0].message.content.strip()

        # Remove markdown fences if present
        if output.startswith("```"):
            lines = output.splitlines()
            lines = [line for line in lines if not line.startswith("```")]
            output = "\n".join(lines)

        result = json.loads(output)

        if "answer" not in result:
            raise ValueError

        if "citations" not in result:
            result["citations"] = []

        if "confidence" not in result:
            result["confidence"] = 0.0

        if "answerable" not in result:
            result["answerable"] = False

        valid_ids = {chunk.chunk_id for chunk in request.chunks}

        result["citations"] = [
            cid for cid in result["citations"] if cid in valid_ids
        ]

        result["confidence"] = max(
            0.0,
            min(1.0, float(result["confidence"]))
        )

        return result

    except Exception:
        return {
            "answer": "I don't know.",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }