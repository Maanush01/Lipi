import os
import csv
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

# Your 25 verified questions
QUESTIONS = [
    ("Kannada sahityada modala kavi yaru?", "Pampa (Adikavi)"),
    ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?", "Bengaluru"),
    ("Who built the Hoysaleswara Temple, and where is it located?", "Hoysala Empire (Vishnuvardhana era); Halebidu"),
    ("ಕರ್ನಾಟಕ ಏಕೀಕರಣ ಯಾವಾಗ ನಡೆಯಿತು?", "1 November 1956"),
    ("Kannada rajyotsava yavaga aacharisalaguthade?", "November 1st"),
    ("'Kavirajamarga' yaaru bareda kruti?", "Attributed to King Nripatunga / poet Srivijaya"),
    ("Who wrote 'Ramayana Darshanam,' and which award did it win?", "Kuvempu; Jnanpith (1967)"),
    ("कन्नड़ भाषा का पहला ज्ञानपीठ पुरस्कार विजेता कौन था?", "Kuvempu"),
    ("Kannadadalli 'Jnanpith Award' pondiruva total ellamandi?", "8"),
    ("Kannada chitrarangada modala talkie chitra yaavudu?", "Sati Sulochana (1934)"),
    ("Dr. Rajkumar avara janma dinaanka mattu ooru yaavudu?", "April 24, 1929; Gajanur"),
    ("Name a Kannada film that won the National Film Award for Best Feature Film.", "e.g., Samskara (1970), Vamsha Vriksha (1971), Thithi (2015)"),
    ("Kannada cinemada 'Crazy Star' endu karayalpaduva nataru yaaru?", "V. Ravichandran"),
    ("ಕರ್ನಾಟಕದ ಅತಿ ಎತ್ತರದ ಶಿಖರ ಯಾವುದು?", "Mullayanagiri"),
    ("Karnatakadalli hariyuva mukhya nadigalu yaavuvu?", "Kaveri, Krishna, Tungabhadra, etc."),
    ("Which Kannada-speaking region is famous for coffee plantations?", "Chikmagalur / Kodagu"),
    ("ನಮಸ್ಕಾರ ಪದದ ಅರ್ಥ ಏನು?", "A greeting/salutation"),
    ("Translate to Kannada: 'How are you?'", "ನೀವು ಹೇಗಿದ್ದೀರಾ?"),
    ("Explain in English: 'Naale nange time illa, busy iddini'", "I don't have time tomorrow, I'm busy"),
    ("What does 'ಗೊತ್ತಿಲ್ಲ' mean?", "I don't know"),
    ("Is there a real difference between 'Karnataka' and 'Kannada Naadu'?", "Largely synonymous; poetic/cultural term"),
    ("Yaava varshadalli Kannada 'Classical Language' status pondithu?", "2008"),
    ("दक्षिण भारत में कन्नड़ भाषा किन राज्यों में बोली जाती है?", "Mainly Karnataka; also Maharashtra, Andhra Pradesh, Tamil Nadu border areas"),
    ("ಕರ್ನಾಟಕದ ರಾಜ್ಯ ಪಕ್ಷಿ ಯಾವುದು?", "Indian Roller (ನೀಲಕಂಠ)"),
    ("Kannada varnamale alli total svara (vowels) ettu?", "13 (traditional count)"),
]


def ask_once(question):
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
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
    for i, (question, expected) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Asking: {question[:50]}...")
        answer = ask_once(question)
        results.append({
            "id": i,
            "question": question,
            "expected": expected,
            "lipi_answer": answer,
            "correct (y/n/partial)": "",  # you'll fill this in
        })
        time.sleep(1)  # small pause between requests, easy on the API

    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "expected", "lipi_answer", "correct (y/n/partial)"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Saved {len(results)} results to eval_results.csv")


if __name__ == "__main__":
    main()