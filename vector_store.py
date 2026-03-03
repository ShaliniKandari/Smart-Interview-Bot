"""
Vector Store for SHL Assessments
Uses TF-IDF embeddings with cosine similarity for retrieval.
Can be upgraded to sentence-transformers when available.
"""

import json
import re
import math
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class TFIDFVectorStore:
    """
    Lightweight TF-IDF based vector store.
    No external ML libraries required.
    """

    def __init__(self):
        self.assessments: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[List[float]] = []
        self.doc_norms: List[float] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer with stopword removal."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "and", "or", "but", "if", "as",
            "it", "its", "this", "that", "these", "those", "i", "you", "we",
            "they", "he", "she", "who", "which", "what", "how", "when", "where",
            "not", "no", "nor", "so", "yet", "both", "either", "neither", "just",
            "also", "more", "most", "other", "some", "such", "than", "too", "very",
        }
        tokens = re.findall(r'\b[a-z][a-z0-9+#\-.]*\b', text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _build_doc_text(self, assessment: Dict) -> str:
        """Build searchable text from assessment fields."""
        parts = []
        
        name = assessment.get("name", "")
        parts.append(name)
        parts.append(name)  # Weight name higher
        
        desc = assessment.get("description", "")
        if desc:
            parts.append(desc)
        
        test_types = assessment.get("test_types", [])
        type_expansions = {
            "A": "ability aptitude cognitive numerical verbal reasoning",
            "B": "biodata situational judgement",
            "C": "competencies behavioral competency",
            "D": "development 360 feedback",
            "E": "assessment exercises simulation",
            "K": "knowledge skills technical proficiency",
            "P": "personality behavior behaviour traits",
            "S": "simulations work sample",
        }
        for t in test_types:
            if t in type_expansions:
                parts.append(type_expansions[t])
        
        return " ".join(parts)

    def build_index(self, assessments: List[Dict]):
        """Build TF-IDF index from assessments."""
        self.assessments = assessments
        docs = [self._build_doc_text(a) for a in assessments]
        
        # Build vocabulary + term frequencies per doc
        tf_per_doc = []
        doc_freqs = defaultdict(int)
        
        for doc in docs:
            tokens = self._tokenize(doc)
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            tf_per_doc.append(dict(tf))
            for token in set(tokens):
                doc_freqs[token] += 1
        
        # Vocabulary
        self.vocab = {word: idx for idx, word in enumerate(sorted(doc_freqs.keys()))}
        n_docs = len(docs)
        
        # IDF
        self.idf = {}
        for word, df in doc_freqs.items():
            self.idf[word] = math.log((n_docs + 1) / (df + 1)) + 1.0
        
        # TF-IDF matrix
        self.tfidf_matrix = []
        self.doc_norms = []
        
        for tf in tf_per_doc:
            total_terms = sum(tf.values()) or 1
            vec = {}
            norm = 0.0
            for word, count in tf.items():
                if word in self.idf:
                    tfidf_val = (count / total_terms) * self.idf[word]
                    vec[word] = tfidf_val
                    norm += tfidf_val ** 2
            self.tfidf_matrix.append(vec)
            self.doc_norms.append(math.sqrt(norm) or 1.0)
        
        print(f"Index built: {len(assessments)} docs, {len(self.vocab)} vocab terms")

    def _query_vector(self, query: str) -> Tuple[Dict[str, float], float]:
        """Compute TF-IDF vector for query."""
        tokens = self._tokenize(query)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        
        total = sum(tf.values()) or 1
        vec = {}
        norm = 0.0
        for word, count in tf.items():
            if word in self.idf:
                v = (count / total) * self.idf[word]
                vec[word] = v
                norm += v ** 2
        return vec, math.sqrt(norm) or 1.0

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return (index, score) pairs sorted by cosine similarity."""
        q_vec, q_norm = self._query_vector(query)
        
        scores = []
        for i, (doc_vec, doc_norm) in enumerate(zip(self.tfidf_matrix, self.doc_norms)):
            dot = sum(q_vec.get(w, 0) * doc_vec.get(w, 0) for w in q_vec)
            cosine = dot / (q_norm * doc_norm)
            scores.append((i, cosine))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str):
        """Persist index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "assessments": self.assessments,
            "vocab": self.vocab,
            "idf": self.idf,
            "tfidf_matrix": self.tfidf_matrix,
            "doc_norms": self.doc_norms,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Index saved to {path}")

    def load(self, path: str):
        """Load persisted index."""
        with open(path) as f:
            data = json.load(f)
        self.assessments = data["assessments"]
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.doc_norms = data["doc_norms"]
        print(f"Index loaded: {len(self.assessments)} assessments")


def build_index_from_file(json_path: str, index_path: str) -> TFIDFVectorStore:
    """Helper to build and save index."""
    with open(json_path) as f:
        assessments = json.load(f)
    
    store = TFIDFVectorStore()
    store.build_index(assessments)
    store.save(index_path)
    return store


if __name__ == "__main__":
    build_index_from_file("data/shl_assessments.json", "data/tfidf_index.json")
