import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = (
    "You are Lipi, a friendly AI assistant. Reply in the same language "
    "the user writes in, including Kannada, Hindi, and Kanglish/Hinglish. "
    "Keep answers clear and helpful."
)

# Tried in order. Check AI Studio for current model names.
MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]

config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
history = []  # we keep the conversation ourselves so we can switch models freely


def ask(user_text):
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=history, config=config
                )
                text = resp.text or "(no response)"
                history.append(types.Content(role="model", parts=[types.Part(text=text)]))
                return text
            except errors.APIError as e:
                print(f"[{model} attempt {attempt + 1} failed: {e.code}]")
                time.sleep(2 ** attempt)  # wait 1s, 2s, 4s
    history.pop()  # remove the unanswered message
    return "Sorry, the servers are busy right now. Please try again in a minute."


print("Lipi is ready. Type 'quit' to exit.")
while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    print("Lipi:", ask(user))