"""
SHL Assessment Recommender
RAG pipeline: TF-IDF retrieval → Anthropic Claude reranking & enrichment
"""

import json
import os
import re
import requests
from typing import List, Dict, Optional
from vector_store import TFIDFVectorStore

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class SHLRecommender:
    def __init__(
        self,
        assessments_path: str = "data/shl_assessments.json",
        index_path: str = "data/tfidf_index.json",
        anthropic_api_key: str = None,
    ):
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.store = TFIDFVectorStore()

        # Load or build index
        if os.path.exists(index_path):
            self.store.load(index_path)
        elif os.path.exists(assessments_path):
            with open(assessments_path) as f:
                assessments = json.load(f)
            self.store.build_index(assessments)
            self.store.save(index_path)
        else:
            raise FileNotFoundError(
                f"Neither {assessments_path} nor {index_path} found. "
                "Run the scraper first."
            )

    def retrieve_candidates(self, query: str, top_k: int = 20) -> List[Dict]:
        """Retrieve top-k candidates using TF-IDF."""
        results = self.store.search(query, top_k=top_k)
        candidates = []
        for idx, score in results:
            a = self.store.assessments[idx].copy()
            a["_retrieval_score"] = round(score, 4)
            candidates.append(a)
        return candidates

    def _parse_duration_constraint(self, query: str) -> Optional[int]:
        """Extract max duration constraint from query."""
        patterns = [
            r'(?:max(?:imum)?|within|less than|under|at most|no more than)\s*(\d+)\s*min',
            r'(\d+)\s*min(?:ute)?s?\s*(?:max|limit|long|duration)',
            r'completed in\s*(\d+)\s*min',
        ]
        query_lower = query.lower()
        for pattern in patterns:
            m = re.search(pattern, query_lower)
            if m:
                return int(m.group(1))
        return None

    def _filter_by_duration(self, candidates: List[Dict], max_duration: int) -> List[Dict]:
        """Filter candidates respecting duration constraint."""
        filtered = [a for a in candidates if a.get("duration") is None or a.get("duration", 999) <= max_duration]
        # If too aggressive, relax slightly
        if len(filtered) < 5:
            filtered = [a for a in candidates if a.get("duration") is None or a.get("duration", 999) <= max_duration + 10]
        return filtered if len(filtered) >= 3 else candidates

    def rerank_with_llm(self, query: str, candidates: List[Dict], n: int = 10) -> List[Dict]:
        """Use Claude to rerank and select best assessments."""
        if not self.api_key:
            # Fallback: return top-n by retrieval score
            return candidates[:n]

        # Prepare candidate list for LLM
        candidate_text = "\n".join([
            f"{i+1}. Name: {c['name']}\n"
            f"   URL: {c['url']}\n"
            f"   Test Types: {', '.join(c.get('test_types', [])) or 'N/A'}\n"
            f"   Duration: {c.get('duration') or 'N/A'} min\n"
            f"   Remote: {c.get('remote_support', 'N/A')} | Adaptive: {c.get('adaptive_support', 'N/A')}\n"
            f"   Description: {c.get('description', '')[:200] or 'N/A'}"
            for i, c in enumerate(candidates[:20])
        ])

        system_prompt = """You are an expert HR assessment consultant for SHL.
Your task is to select the most relevant assessments for a given query.

Rules:
1. Select 5–10 assessments that best match the query requirements
2. Balance hard skills (Test Type K) and soft skills/personality (Test Type P/C) when query mentions both
3. Respect any duration constraints mentioned in the query
4. Respond ONLY with a JSON array of numbers (1-indexed positions from the candidate list)
   Example: [1, 3, 5, 7, 2]
3. Do not include any explanation, just the JSON array."""

        user_prompt = f"""Query: {query}

Candidates:
{candidate_text}

Select the best 5-10 assessments. Return ONLY a JSON array of their numbers (e.g., [1, 4, 7, 2, 9])."""

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"].strip()
            
            # Extract JSON array
            match = re.search(r'\[[\d,\s]+\]', content)
            if match:
                indices = json.loads(match.group())
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(candidates):
                        selected.append(candidates[idx - 1])
                if selected:
                    return selected[:n]
        except Exception as e:
            print(f"LLM reranking failed: {e}, using retrieval scores")
        
        return candidates[:n]

    def recommend(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Full recommendation pipeline:
        1. Retrieve candidates via TF-IDF
        2. Filter by duration constraint if present
        3. Rerank with LLM
        4. Format output
        """
        # Step 1: Retrieve
        candidates = self.retrieve_candidates(query, top_k=25)
        
        # Step 2: Filter duration
        max_dur = self._parse_duration_constraint(query)
        if max_dur:
            candidates = self._filter_by_duration(candidates, max_dur)
        
        # Step 3: LLM rerank
        selected = self.rerank_with_llm(query, candidates, n=n_results)
        
        # Step 4: Format
        results = []
        for a in selected:
            results.append({
                "name": a.get("name", ""),
                "url": a.get("url", ""),
                "description": a.get("description", ""),
                "duration": a.get("duration"),
                "remote_support": a.get("remote_support", "No"),
                "adaptive_support": a.get("adaptive_support", "No"),
                "test_type": a.get("test_types", []),
            })
        
        return results[:n_results]


if __name__ == "__main__":
    # Quick test
    rec = SHLRecommender()
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    results = rec.recommend(query)
    print(f"\nResults for: {query}\n")
    for r in results:
        print(f"  - {r['name']}: {r['url']}")
