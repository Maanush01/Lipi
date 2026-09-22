# Lipi

A friendly AI assistant for Kannada, Hindi, and Kanglish, built on the Gemini API, with a Retrieval-Augmented Generation (RAG) pipeline for verified factual accuracy on niche topics.

**Live demo:** [https://lipi-kannada.streamlit.app](https://lipi-kannada.streamlit.app)

## Overview

Lipi went through three iterations as I learned to build, evaluate, and improve an LLM-powered app:

1. **Terminal version** (`terminal-version/`) — a command-line chatbot with retry logic and model fallback
2. **Gradio version** (`gradio-version/`) — a local chat UI built with Gradio
3. **Streamlit version** (`streamlit_app.py`) — the deployed, public version

Beyond the chatbot itself, this project includes a full **evaluation pipeline** to measure and improve factual accuracy, rather than just assuming the model "seems to work."

## Features

- Replies in the user's language and script (Kannada, Hindi, Kanglish/Hinglish)
- Streaming responses for fast time-to-first-token
- Retry with exponential backoff and model fallback for API errors
- A custom evaluation suite measuring factual accuracy
- A RAG pipeline that grounds answers in verified source documents

## Evaluation & Results

To move beyond "it seems to work," I built two evaluation sets and measured Lipi's accuracy on each, before and after adding retrieval-augmented generation.

### 1. General knowledge (25 questions)

A broad set of well-known Kannada history, literature, cinema, geography, and language questions, verified against reliable sources before grading.

**Result: 24/25 correct (96%)**

The one miss was a subtle conflation between two related facts (the _oldest surviving Kannada text_ vs. the _traditionally honored first poet_), not a random hallucination, showing that Gemini's baseline knowledge on mainstream Kannada topics is already strong.

### 2. Niche knowledge — Akka Mahadevi & Vachana Sahitya (18 questions)

To find where the base model's knowledge genuinely breaks down, I built a harder, narrower question set on a specific 12th-century Kannada literary movement, with facts (exact names, dates, places, book titles) unlikely to be well-represented in general training data.

|                              | Without RAG | With RAG         |
| ---------------------------- | ----------- | ---------------- |
| **Accuracy**                 | 8/18 (44%)  | **18/18 (100%)** |
| **Confident hallucinations** | 4 questions | 0                |

Examples of baseline failures the RAG pipeline fixed:

- Confused a historical figure (King Kaushika) with an unrelated Vedic sage (Vishvamitra), inventing an entire wrong backstory
- Fabricated three plausible-sounding but incorrect "marriage conditions"
- Got a discovery date, location, and century all wrong for a historical artifact
- Could not answer a specific citation question at all

### How the RAG pipeline works

1. A source document (`rag/sourceDocument.md`) is split into chunks
2. Each chunk is embedded using Google's `gemini-embedding-001` model and cached locally (`rag/chunk_Index.json`)
3. At query time, the question is embedded and compared against all chunks using cosine similarity
4. The top matching chunks are injected into the prompt as context before Gemini generates an answer
5. The system prompt instructs the model to answer _only_ from the provided context, and say so if the context doesn't contain the answer

This is a lightweight, dependency-free RAG implementation (no external vector database), suitable for small, focused knowledge bases.

## Tech stack

- Python
- Google Gemini API (`google-genai`) — chat + embeddings (`gemini-embedding-001`)
- Streamlit
- (Earlier versions: Gradio, plain CLI)

## Run locally

1. `pip install -r requirements.txt`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. `python -m streamlit run streamlit_app.py`

## Run the evaluations

​`
python "eval/eval.py"                     # general knowledge (25 questions)
python "evel hard/eval_hard.py"           # niche knowledge baseline, no RAG
python "eval hard Rag/eval_hard_rag.py"   # niche knowledge, with RAG
​`

All three scripts save results as CSVs for manual grading.
