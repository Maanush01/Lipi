# Lipi

A friendly AI assistant for Kannada, Hindi, and Kanglish, built with the Gemini API.

Lipi went through three versions as I learned to build and deploy an LLM-powered app:

1. **Terminal version** (`terminal-version/`) — a command-line chatbot with retry logic and model fallback
2. **Gradio version** (`gradio-version/`) — a local chat UI built with Gradio
3. **Streamlit version** (`streamlit_app.py`) — the deployed version, live at [link]

## Features

- Replies in the user's language and script (Kannada, Hindi, Kanglish/Hinglish)
- Streaming responses for fast time-to-first-token
- Retry with exponential backoff and model fallback for API errors

## Live demo

[https://lipi-yourname.streamlit.app](https://lipi-yourname.streamlit.app)

## Run locally

1. `pip install -r requirements.txt`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. `python -m streamlit run streamlit_app.py`

## Tech stack

- Python
- Google Gemini API (`google-genai`)
- Streamlit
- (Earlier versions: Gradio, plain CLI)
