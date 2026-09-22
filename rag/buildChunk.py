import os
from dotenv import load_dotenv
from google import genai
from rag_utils import build_index

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
build_index(client, "sourceDocument.md")