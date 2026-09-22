import os
import csv
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from rag_utils import retrieve

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = (
    "You are Lipi, a friendly AI assistant. You will be given some CONTEXT "
    "retrieved from a knowledge base, followed by a QUESTION. Answer the "
    "question using ONLY the information in the CONTEXT. If the context "
    "doesn't contain the answer, say you don't have enough information, "
    "instead of guessing. Reply in the same language and script the user "
    "writes in. Keep replies short (under 150 words) unless asked for detail."
)

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

# Same 18 hard questions as eval_hard.py
HARD_QUESTIONS = [
    ("In which village was Akka Mahadevi born, and near which larger town?",
     "Born in Udutadi, near Shivamogga, Karnataka, around 1130."),
    ("What were the names of Akka Mahadevi's parents?",
     "Nirmalshetti and Sumati (also given as Nirmal Shetti and Sumati Shetty)."),
    ("How many vachanas is Akka Mahadevi credited with, according to the most commonly cited figure?",
     "430 vachanas (some sources say ~350, based on tracing her ankita)."),
    ("What are the names of the two short prose works (not vachanas) attributed to Akka Mahadevi?",
     "Mantrogopya and Yogangatrividhi."),
    ("What is Akka Mahadevi's 'ankita' (signature name for her chosen deity), and what does it mean?",
     "Chennamallikarjuna, referring to Shiva; translated by A.K. Ramanujan as 'Lord, white as jasmine.'"),
    ("Who translated Akka Mahadevi's vachanas into English in a collection titled 'Speaking of Siva'?",
     "A. K. Ramanujan."),
    ("Which scholar critiqued that translation, and in which book?",
     "Tejaswini Niranjana, in 'Siting Translation' (1992)."),
    ("What conditions did Akka Mahadevi reportedly lay down before agreeing to marry King Kaushika?",
     "Control over her time to spend in devotion or conversation with scholars/religious figures, rather than being required to spend it with the king."),
    ("What religious community did King Kaushika belong to?",
     "Jain."),
    ("According to Harihara's account, what did Akka Mahadevi do when the king violated the conditions?",
     "She left the palace, renouncing all possessions including clothes, and travelled to Srisailam."),
    ("What are the three phases of Akka Mahadevi's spiritual life, as traditionally described?",
     "1) Renouncing worldly objects/attractions, 2) discarding object-based rules/regulations, 3) journeying toward Srisaila."),
    ("At what specific forest location within Srisailam is Akka Mahadevi said to have attained union with Shiva?",
     "Kadali (the dense forest area of Srisailam)."),
    ("What term is used for the philosophical assembly hall in Kalyana, and who gave Akka Mahadevi her honorific title?",
     "Anubhava Mantapa; saints including Basavanna, Chenna Basavanna, Kinnari Bommayya, Siddharama, Allama Prabhu, and Dasimayya."),
    ("Approximately how many Vachana writers (Vachanakaras) have been historically recorded, and how many were women?",
     "More than 200 recorded, more than 30 of whom were women."),
    ("Where and when was a bas-relief believed to depict Akka Mahadevi discovered?",
     "Near Hospet, Karnataka, discovered in 2010 (dating to the 13th century); some sources also mention a Chitradurga find."),
    ("What role did Allama Prabhu play in the movement's philosophy, as distinct from Basavanna's role?",
     "Allama provided the philosophical and mystical core; Basavanna forged the social philosophy of the movement."),
    ("In which district is Basavakalyana located?",
     "Bidar district."),
    ("What does the word 'vachana' literally mean?",
     "'(That which is) said.'"),
]


def ask_with_rag(question):
    chunks = retrieve(client, question, top_k=3)
    context = "\n\n---\n\n".join(chunks)
    full_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    contents = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]
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
    return f"ERROR: {last_error}"


def main():
    results = []
    for i, (question, expected) in enumerate(HARD_QUESTIONS, 1):
        print(f"[{i}/{len(HARD_QUESTIONS)}] Asking (with RAG): {question[:60]}...")
        answer = ask_with_rag(question)
        results.append({
            "id": i,
            "question": question,
            "expected": expected,
            "lipi_answer_rag": answer,
            "correct (y/n/partial)": "",
        })
        time.sleep(1)

    with open("eval_hard_rag.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "expected", "lipi_answer_rag", "correct (y/n/partial)"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Saved {len(results)} results to eval_hard_rag.csv")


if __name__ == "__main__":
    main()