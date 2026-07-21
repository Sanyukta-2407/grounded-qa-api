import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)


class Chunk(BaseModel):
    chunk_id: str
    text: str


class QARequest(BaseModel):
    question: str
    chunks: list[Chunk]


@app.post("/")
def grounded_qa(request: QARequest):
    if not request.question.strip():
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    if len(request.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    context = ""

    for chunk in request.chunks:
        context += f"{chunk.chunk_id}: {chunk.text}\n"

    prompt = f"""
You are a grounded question answering system.

Answer ONLY using the supplied context.

If the answer is not explicitly supported by the context, return:

{{
"answer":"I don't know",
"citations":[],
"confidence":0.2,
"answerable":false
}}

If answerable, return ONLY valid JSON:

{{
"answer":"...",
"citations":["C1"],
"confidence":0.95,
"answerable":true
}}

Use only provided chunk IDs.

Context:
{context}

Question:
{request.question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = json.loads(response.choices[0].message.content)

        valid_ids = {c.chunk_id for c in request.chunks}

        result["citations"] = [
            cid for cid in result.get("citations", [])
            if cid in valid_ids
        ]

        if not result.get("answerable", False):
            result["answer"] = "I don't know"
            result["citations"] = []
            result["confidence"] = min(
                float(result.get("confidence", 0.2)),
                0.3,
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))