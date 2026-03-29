from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator
from starlette.requests import Request
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
import re
import logging
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Env / startup validation
load_dotenv()

_REQUIRED = ["API_KEY", "HF_API_TOKEN", "PROD_ORIGIN"]
_missing = [k for k in _REQUIRED if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(_missing)}")

API_KEY  = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
HF_URL   = "https://router.huggingface.co/v1"

# CORS
environment = os.getenv("ENVIRONMENT", "production")
if environment == "development":
    dev_origin = os.getenv("DEV_ORIGIN")
    if not dev_origin:
        raise RuntimeError("DEV_ORIGIN must be set in development mode")
    origins = [dev_origin]
else:
    origins = [os.getenv("PROD_ORIGIN")]

# App + rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["x-api-key", "content-type"],
)

# Grounding context
def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

def build_context(projects: list) -> str:
    lines = []
    for p in projects:
        lines.append(f"Project: {p.get('name')}")
        lines.append(f"Slug: {p.get('slug')}")
        stack = p.get("data", {}).get("stack", [])
        if stack:
            lines.append(f"Stack: {', '.join(stack)}")
        for url_group in p.get("data", {}).get("projectUrls", []):
            for item in url_group:
                if isinstance(item, list):
                    lines.append(f"URL: {item[1]} → {item[0]}")
        content = _strip_html(p.get("article", {}).get("content", ""))
        lines.append(f"Description: {content}")
        lines.append("---")
    return "\n".join(lines)

try:
    with open("projects.json", "r", encoding="utf-8") as f:
        projects = json.load(f)
    CONTEXT = build_context(projects)
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(f"Failed to load projects.json: {e}")

# LangChain chain
SYSTEM_PROMPT = f"""Du bist ein Assistent auf dem Portfolio von Benedikt Martini (Information Designer & Entwickler Interaktive Datenvisualisierungen).
Besucher der Seite stellen dir Fragen zu seinen Projekten, Tools und Fachbereichen.
Antworte immer aus der Perspektive des Portfolios — nicht als Benedikt selbst, aber auch nicht als externer Beobachter.

Regeln:

- Basis sind ausschließlich die Projektdaten unten
- Schlüsse aus den Daten sind erlaubt (z.B. Tool-Häufigkeiten zählen, verwendete Modelle aus Beschreibungen ableiten)
- Keine Spekulation über Dinge die nicht in den Daten stehen
- Kein Erklären deines Denkprozesses — nur das Ergebnis
- Slugs nie im Text verwenden, nur in URLs
- URLs nur wenn explizit gefragt, Format: <a href="https://benediktmartini.de/projects/[slug]" target="_blank" style="color:#252526">Projektname</a>
- Wenn etwas nicht in den Daten steht: "Dazu habe ich keine Informationen."
- Sprich nicht über Benedikt, nur über die Projekte
--- PROJEKTDATEN ---
{CONTEXT}
--- ENDE ---"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{message}"),
])

llm = ChatOpenAI(
    model=HF_MODEL,
    openai_api_key=HF_TOKEN,
    openai_api_base=HF_URL,
    max_tokens=512,
)

chain = prompt | llm | StrOutputParser()

# Request model
class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > 1000:
            raise ValueError("message must not exceed 1000 characters")
        return v

# Auth
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Route
@app.post("/portfolio-chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest, _: str = Depends(verify_api_key)):
    try:
        reply = await chain.ainvoke({"message": req.message})
    except Exception as e:
        logger.error("LangChain chain error: %s", e)
        raise HTTPException(status_code=502, detail="Upstream request failed")

    return {"reply": reply}