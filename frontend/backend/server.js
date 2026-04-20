require('dotenv').config();
const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
app.use(cors());
app.use(express.json());

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const DOMAIN_PROMPTS = {
  dsa: 'Data Structures & Algorithms (arrays, trees, graphs, sorting, dynamic programming, time/space complexity)',
  hr: 'HR & Behavioral (STAR method, leadership, conflict resolution, teamwork, career goals)',
  system: 'System Design (scalability, microservices, databases, caching, load balancing, APIs)',
  frontend: 'Frontend Development (React, JavaScript, CSS, browser APIs, performance, accessibility)',
};

// Generate a question
app.post('/api/question', async (req, res) => {
  const { domain, questionNumber, totalQuestions, previousQuestions = [] } = req.body;
  const domainLabel = DOMAIN_PROMPTS[domain] || domain;

  try {
    const msg = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 300,
      system: `You are a professional technical interviewer for ${domainLabel}.
Ask one clear, focused interview question. No preamble, no numbering, just the question.
Avoid repeating these questions: ${previousQuestions.join(' | ')}`,
      messages: [{ role: 'user', content: `Question ${questionNumber} of ${totalQuestions}.` }],
    });
    res.json({ question: msg.content[0].text.trim() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Evaluate an answer
app.post('/api/evaluate', async (req, res) => {
  const { domain, question, answer, questionNumber, totalQuestions } = req.body;
  const domainLabel = DOMAIN_PROMPTS[domain] || domain;
  const isLast = questionNumber >= totalQuestions;

  try {
    const msg = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 600,
      system: `You are an expert interviewer evaluating a ${domainLabel} interview answer.
Return ONLY valid JSON with no markdown fences. Schema:
{
  "confidence": <0-100>,
  "keywords": <0-100>,
  "clarity": <0-100>,
  "feedback": "<1-2 sentence constructive feedback>",
  "nextQuestion": "<next interview question string, or null if last question>"
}
isLastQuestion: ${isLast}`,
      messages: [{
        role: 'user',
        content: `Question ${questionNumber}/${totalQuestions}: ${question}\n\nAnswer: ${answer}`,
      }],
    });

    const raw = msg.content[0].text.replace(/```json|```/g, '').trim();
    const result = JSON.parse(raw);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get('/health', (_, res) => res.json({ status: 'ok' }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
