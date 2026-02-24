from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://benediktmartini.de"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# Load and flatten projects.json as grounding context
with open("projects.json", "r", encoding="utf-8") as f:
    projects = json.load(f)

def build_context(projects):
    lines = []
    for p in projects:
        lines.append(f"Project: {p.get('name')}")
        lines.append(f"Slug: {p.get('slug')}")
        stack = p.get("data", {}).get("stack", [])
        if stack:
            lines.append(f"Stack: {', '.join(stack)}")
        urls = p.get("data", {}).get("projectUrls", [])
        for url_group in urls:
            for item in url_group:
                if isinstance(item, list):
                    lines.append(f"URL: {item[1]} → {item[0]}")
        content = p.get("article", {}).get("content", "")
        # strip html tags roughly
        import re
        content = re.sub(r"<[^>]+>", "", content)
        lines.append(f"Description: {content}")
        lines.append("---")
    return "\n".join(lines)

CONTEXT = build_context(projects)

SYSTEM_PROMPT = f"""Du bist der Portfolio-Assistent dieser Portfolioseite (Information Design, 
Data Visualization kombiniert mit Web Development).
Beantworte Fragen zu Projekten, Tools und Themen ausschließlich auf Basis der Projektdaten unten.
Keine Spekulationen, keine Erklärungen deines Denkprozesses. Stelle aber Rechnungen auf Basis der Daten an: 
Wenn der User bspw. fragt welches Tool am meisten benutzt wird, errechne die Summe anhand der Nennungen in stack.
Wenn etwas nicht in den Daten steht, sage nur: "Dazu habe ich keine Informationen."

Projekt-URLs haben folgende Struktur: https://benediktmartini.de/projects/[slug]
Füge URLs nur hinzu wenn der Nutzer explizit danach fragt.
Verwende niemals Slugs im Text, nur in den URLs.
URLs immer als: <a href="URL" target="_blank" style="color:#252526">Linktext</a>

--- PROJEKTDATEN ---
{CONTEXT}
--- ENDE ---"""


class ChatRequest(BaseModel):
    message: str


@app.post("/portfolio-chat")
async def chat(req: ChatRequest):
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nUser question: {req.message} [/INST]"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "model": f"{HF_MODEL}:cerebras",  # cerebras is fast and free
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.message}
                ],
                "max_tokens": 512
            },
        )
    print("STATUS:", response.status_code)
    print("BODY:", response.text)

    if response.status_code != 200:
        return {"reply": f"HF API error: {response.status_code} - {response.text}"}

    result = response.json()
    print("result:", result)
    reply = result["choices"][0]["message"]["content"].strip()
    print(reply)
    return {"reply": reply}
