"""
Optional Python NLP microservice for deeper answer analysis.
Uses spaCy for keyword extraction and sentence scoring.

Install: pip install fastapi uvicorn spacy
         python -m spacy download en_core_web_sm

Run: uvicorn nlp_service:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import spacy
import re

app = FastAPI(title="Interview NLP Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")

# Domain keyword banks
DOMAIN_KEYWORDS = {
    "dsa": [
        "array", "linked list", "tree", "graph", "hash", "stack", "queue",
        "heap", "binary", "recursion", "dynamic programming", "memoization",
        "big o", "complexity", "time complexity", "space complexity",
        "sorting", "searching", "bfs", "dfs", "greedy", "divide and conquer",
    ],
    "hr": [
        "team", "leadership", "communication", "conflict", "resolution",
        "collaboration", "deadline", "priority", "challenge", "outcome",
        "result", "initiative", "feedback", "improvement", "stakeholder",
        "situation", "action", "task", "star method",
    ],
    "system": [
        "scalability", "microservices", "database", "cache", "load balancer",
        "api", "rest", "latency", "throughput", "availability", "partition",
        "consistency", "replication", "sharding", "cdn", "message queue",
        "kafka", "redis", "sql", "nosql", "cap theorem",
    ],
    "frontend": [
        "react", "component", "state", "props", "hook", "useeffect",
        "dom", "virtual dom", "css", "flexbox", "grid", "responsive",
        "accessibility", "performance", "lazy loading", "typescript",
        "webpack", "bundler", "api", "fetch", "async", "promise",
    ],
}


class AnalyzeRequest(BaseModel):
    answer: str
    domain: str
    question: str = ""


class AnalyzeResponse(BaseModel):
    keyword_score: int
    clarity_score: int
    confidence_score: int
    matched_keywords: List[str]
    word_count: int
    sentence_count: int
    feedback: str


def extract_keywords(text: str, domain: str) -> List[str]:
    lower = text.lower()
    bank = DOMAIN_KEYWORDS.get(domain, [])
    return [kw for kw in bank if kw in lower]


def score_clarity(doc) -> int:
    """Score based on sentence structure and length balance."""
    sents = list(doc.sents)
    if not sents:
        return 0
    avg_len = sum(len(s) for s in sents) / len(sents)
    # Ideal sentence length: 10-25 tokens
    if 10 <= avg_len <= 25:
        base = 85
    elif avg_len < 5:
        base = 40
    elif avg_len > 40:
        base = 60
    else:
        base = 70
    # Bonus for multiple sentences (shows structure)
    if len(sents) >= 3:
        base = min(100, base + 10)
    return base


def score_confidence(text: str, doc) -> int:
    """Score based on use of hedging vs assertive language."""
    hedge_words = ["maybe", "i think", "not sure", "possibly", "perhaps", "i guess", "kind of", "sort of"]
    strong_words = ["definitely", "clearly", "because", "therefore", "specifically", "for example", "in conclusion"]
    lower = text.lower()
    hedge_count = sum(1 for w in hedge_words if w in lower)
    strong_count = sum(1 for w in strong_words if w in lower)
    word_count = len(text.split())
    base = 60
    base -= hedge_count * 8
    base += strong_count * 6
    if word_count > 80:
        base += 10  # detailed answers are more confident
    return max(20, min(100, base))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_answer(req: AnalyzeRequest):
    doc = nlp(req.answer)
    matched = extract_keywords(req.answer, req.domain)
    keyword_score = min(100, len(matched) * 12 + 30) if matched else 20
    clarity_score = score_clarity(doc)
    confidence_score = score_confidence(req.answer, doc)

    word_count = len(req.answer.split())
    sentence_count = len(list(doc.sents))

    tips = []
    if keyword_score < 50:
        tips.append("Try using more domain-specific terms.")
    if word_count < 40:
        tips.append("Expand your answer with more detail.")
    if not tips:
        tips.append("Good use of relevant concepts and structure.")
    feedback = " ".join(tips)

    return AnalyzeResponse(
        keyword_score=keyword_score,
        clarity_score=clarity_score,
        confidence_score=confidence_score,
        matched_keywords=matched,
        word_count=word_count,
        sentence_count=sentence_count,
        feedback=feedback,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
