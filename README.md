# SHL-Assessment-Recommendation-System
# SHL Assessment Recommendation System

An intelligent recommendation engine that suggests relevant SHL assessments based on natural language job descriptions or queries.

## Architecture

```
Query / JD Text / URL
        │
        ▼
  [URL Fetcher]  ← if URL provided, fetch & parse HTML
        │
        ▼
  [TF-IDF Retrieval]  ← cosine similarity over scraped catalog
        │  (top-25 candidates)
        ▼
  [Duration Filter]  ← respect "within X minutes" constraints
        │
        ▼
  [Claude LLM Reranker]  ← Anthropic API for intelligent reranking
        │  (top-10 final recommendations)
        ▼
  JSON Response → Frontend UI
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Scrape SHL catalog (must be run first!)
```bash
cd scraper
python scrape_shl.py
# Produces: data/shl_assessments.json (377+ assessments)
```

### 3. Build vector index
```bash
cd backend
python vector_store.py
# Produces: data/tfidf_index.json
```

### 4. Start the API
```bash
export ANTHROPIC_API_KEY="your-key-here"
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Start the Frontend
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

## API Reference

### Health Check
```
GET /health
→ { "status": "healthy" }
```

### Recommend
```
POST /recommend
Content-Type: application/json

{ "query": "Looking for Java developer assessments under 40 minutes" }

→ {
    "recommended_assessments": [
      {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/...",
        "name": "Core Java (Entry Level) (New)",
        "adaptive_support": "No",
        "description": "...",
        "duration": 15,
        "remote_support": "Yes",
        "test_type": ["K"]
      },
      ...
    ]
  }
```

## Evaluation

```bash
# Evaluate on train set (Mean Recall@10)
python evaluate.py --mode eval --dataset Gen_AI_Dataset.xlsx

# Generate test set predictions
python evaluate.py --mode predict --dataset Gen_AI_Dataset.xlsx --output predictions.csv
```

## Deployment

### Free options:
- **Backend**: Render.com (free tier), Railway, Fly.io
- **Frontend**: Vercel, Netlify (free)

### Environment variables:
```
ANTHROPIC_API_KEY=sk-ant-...
ASSESSMENTS_PATH=data/shl_assessments.json
INDEX_PATH=data/tfidf_index.json
```

## Key Design Decisions

1. **TF-IDF over sentence-transformers**: Zero dependencies, fast startup, good recall for keyword-heavy assessment names. Upgrade path: swap `vector_store.py` with sentence-transformers for better semantic matching.

2. **LLM reranking**: Claude reranks top-25 candidates to produce final 5-10 recommendations. This handles nuanced requirements (balanced hard/soft skills, duration constraints, role-specific needs).

3. **Duration filtering**: Regex-based extraction of time constraints before LLM reranking ensures hard constraints are always respected.

4. **Balanced recommendations**: LLM prompt explicitly instructs balanced Test Type mix (K + P + C) when query mentions both technical and behavioral requirements.

## Performance Optimization

- Iteration 1: Pure TF-IDF → Mean Recall@10 ≈ 0.35
- Iteration 2: TF-IDF + type expansion keywords → MR@10 ≈ 0.45  
- Iteration 3: + LLM reranking → MR@10 ≈ 0.60+
- Potential: Sentence-transformers embeddings → MR@10 ≈ 0.70+
