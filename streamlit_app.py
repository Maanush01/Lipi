import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it under Settings → Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = (
    "You are Lipi, a friendly AI assistant. Reply in the same language and "
    "script the user writes in, including Kannada, Hindi, and Kanglish/Hinglish. "
    "If you are not sure about a fact, say so instead of guessing. Keep replies "
    "short (under 150 words) unless the user asks for detail."
)

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

st.set_page_config(page_title="Lipi", page_icon="💬")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("Lipi")
st.caption("A friendly assistant for Kannada, Hindi, and Kanglish.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def build_contents():
    contents = []
    for m in st.session_state.messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return contents


def ask_lipi():
    contents = build_contents()
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                return resp.text or "(no response)"
            except errors.APIError as e:
                last_error = e
                if e.code not in (429, 500, 503):
                    break
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                break
    return f"Sorry, Lipi couldn't reach Gemini right now.\n\nLast error: {last_error}"


if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Lipi is thinking..."):
            reply = ask_lipi()
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})