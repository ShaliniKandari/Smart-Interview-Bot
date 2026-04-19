# Smart Interview Bot

AI-powered interview practice with domain-specific questions, speech-to-text, and real-time feedback scoring.

---

## Project Structure

```
smart-interview-bot/
├── frontend/          # React + Vite + Tailwind UI
│   └── src/
│       ├── components/    # ScoreCard
│       ├── hooks/         # useSpeech (speech-to-text)
│       ├── pages/         # SetupScreen, InterviewScreen, ResultsScreen
│       ├── store.js       # Zustand global state
│       ├── api.js         # Axios API client
│       └── App.jsx
├── backend/           # Node.js + Express API
│   ├── server.js
│   ├── package.json
│   └── .env.example
└── ml/                # Optional Python NLP microservice
    ├── nlp_service.py
    └── requirements.txt
```

---

## Quick Start

### 1. Backend

```bash
cd backend
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
npm install
npm run dev
# Runs on http://localhost:3001
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### 3. (Optional) Python NLP Microservice

```bash
cd ml
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn nlp_service:app --port 8000
```

---

## Features

- 4 interview domains: DSA, HR & Behavioral, System Design, Frontend Dev
- 5 questions per session generated live by Claude AI
- Real-time scoring: Confidence, Keywords, Clarity (0–100)
- Constructive feedback after each answer
- Speech-to-text via browser Web Speech API (no API key needed)
- Session summary with overall grade
- Zustand state management
- Proxy config — no CORS issues in dev

---

## Environment Variables

### Backend `.env`
```
ANTHROPIC_API_KEY=sk-ant-...
PORT=3001
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/question` | Generate an interview question |
| POST | `/api/evaluate` | Evaluate an answer, return scores + next question |
| GET  | `/health` | Health check |

### POST `/api/question`
```json
{
  "domain": "dsa",
  "questionNumber": 1,
  "totalQuestions": 5,
  "previousQuestions": []
}
```

### POST `/api/evaluate`
```json
{
  "domain": "dsa",
  "question": "Explain binary search.",
  "answer": "Binary search works by...",
  "questionNumber": 1,
  "totalQuestions": 5
}
```

Response:
```json
{
  "confidence": 78,
  "keywords": 85,
  "clarity": 72,
  "feedback": "Good explanation. Add time complexity analysis.",
  "nextQuestion": "What is the time complexity of quicksort?"
}
```

---

## Deployment

| Service | What |
|---------|------|
| [Vercel](https://vercel.com) | Frontend (React) |
| [Railway](https://railway.app) | Backend (Node.js) |
| [Render](https://render.com) | Alternative backend host |
| [Fly.io](https://fly.io) | Python NLP microservice |

Set `VITE_API_URL=https://your-backend.railway.app` in frontend `.env` and update `api.js` baseURL accordingly.

---

## Tech Stack

- **Frontend:** React 18, Vite, Tailwind CSS, Zustand, Axios
- **Backend:** Node.js, Express, Anthropic SDK
- **AI:** Claude (claude-opus-4-5) via Anthropic API
- **Speech:** Web Speech API (browser-native)
- **NLP (optional):** Python, FastAPI, spaCy
