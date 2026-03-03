"""
SHL Assessment Recommendation API
FastAPI backend with /health and /recommend endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import requests
from bs4 import BeautifulSoup
import re

sys.path.insert(0, os.path.dirname(__file__))
from recommender import SHLRecommender

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Recommends SHL assessments based on job descriptions or queries",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender (lazy loaded)
_recommender: Optional[SHLRecommender] = None

def get_recommender() -> SHLRecommender:
    global _recommender
    if _recommender is None:
        _recommender = SHLRecommender(
            assessments_path=os.environ.get("ASSESSMENTS_PATH", "data/shl_assessments.json"),
            index_path=os.environ.get("INDEX_PATH", "data/tfidf_index.json"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    return _recommender


# ── Request/Response Models ──────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Recommend relevant SHL assessments for a query or job description.
    
    - query: Natural language query or job description text (or URL to JD)
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Handle URL input — fetch page text
    if query.startswith("http://") or query.startswith("https://"):
        try:
            resp = requests.get(query, timeout=10, headers={
                "User-Agent": "Mozilla/5.0"
            })
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove script/style
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            query = soup.get_text(" ", strip=True)[:3000]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")
    
    if len(query) < 5:
        raise HTTPException(status_code=400, detail="Query too short")
    
    try:
        rec = get_recommender()
        results = rec.recommend(query, n_results=10)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Assessment data not loaded. {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Ensure min 1 result
    if not results:
        raise HTTPException(status_code=404, detail="No relevant assessments found")
    
    return RecommendResponse(recommended_assessments=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
