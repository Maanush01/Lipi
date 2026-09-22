import os
import time
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = (
    "You are Lipi, a friendly AI assistant. Reply in the same language "
    "and script the user writes in, including Kannada, Hindi, and "
    "Kanglish/Hinglish. If you are not sure about a fact, say so instead "
    "of guessing. Keep answers clear and helpful."
)

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    thinking_config=types.ThinkingConfig(thinking_budget=0),  # turn off silent reasoning
)

def build_contents(message, history):
    contents = []
    for turn in history:
        if isinstance(turn, dict):  # newer Gradio format
            role = "model" if turn["role"] == "assistant" else "user"
            text = turn["content"]
            if not isinstance(text, str):
                continue
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        else:  # older (user, bot) pair format
            user_msg, bot_msg = turn
            contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))
            if bot_msg:
                contents.append(types.Content(role="model", parts=[types.Part(text=bot_msg)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    return contents

def lipi(message, history):
    contents = build_contents(message, history)
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                text = ""
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        text += chunk.text
                        yield text  # Gradio updates the chat as text arrives
                if text:
                    return
            except errors.APIError as e:
                last_error = e
                print(f"[{model} attempt {attempt + 1}] {e.code}: {e}")
                if e.code not in (429, 500, 503):
                    break
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                print(f"[{model}] unexpected error: {e}")
                break
    yield f"Sorry, Lipi couldn't reach Gemini right now.\n\nLast error: {last_error}"


demo = gr.ChatInterface(
    fn=lipi,
    title="Lipi",
    description="A friendly assistant for Kannada, Hindi, and Kanglish.",
)

if __name__ == "__main__":
    demo.launch()