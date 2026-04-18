import os

import requests
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
FALLBACK_MODELS = [
    GROQ_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]


def _dedupe(values):
    seen = set()
    deduped = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _list_groq_models(api_key: str):
    response = requests.get(
        GROQ_MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Groq model list failed with HTTP {response.status_code}: {response.text[:250]}"
        )

    payload = response.json()
    return [model.get("id") for model in payload.get("data", []) if model.get("id")]


def _select_groq_model():
    if not GROQ_API_KEY:
        raise RuntimeError("Missing required environment variable: GROQ_API_KEY")

    candidates = _dedupe(FALLBACK_MODELS)
    try:
        available_models = _list_groq_models(GROQ_API_KEY)
        available_set = set(available_models)
        for candidate in candidates:
            if candidate in available_set:
                return candidate
        if available_models:
            return available_models[0]
    except Exception:
        pass

    return candidates[0]


def get_llm_chain(retriever):
    model_name = _select_groq_model()
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are MediBot, a medical-document assistant.
Use only the provided context and never invent facts.

Context:
{context}

User question:
{question}

Answer rules:
- Be clear, calm, and concise.
- If context is insufficient, say you could not find relevant information in the provided documents.
- Do not diagnose diseases.
- Do not prescribe medications, doses, or treatment plans.
- Do not say that the context is "mainly about" any specific disease.
- Avoid naming unrelated diseases unless the user directly asks about them.
- Always use this exact sectioned format in markdown:
  ## Possible causes
  ## Red flags
  ## Next steps
  ## When to seek urgent care
- For symptom questions, fill each section using context-only evidence.
- In "Possible causes", list only possibilities from context (not diagnosis).
- Use uncertainty language explicitly: "could", "may", "might", "depends on", "cannot confirm from chat alone".
- Do not provide numeric probabilities or percentages for diagnoses.
- If ranking possibilities, use qualitative labels only: "Higher possibility", "Moderate possibility", "Lower possibility".
- In "Red flags", include warning signs relevant to the user question.
- In "Next steps", give safe non-prescription guidance.
- In "When to seek urgent care", clearly state emergency triggers.
- Never reassure high-risk chest pain + breathlessness + sweating as anxiety-only; always advise urgent emergency evaluation.
""",
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
