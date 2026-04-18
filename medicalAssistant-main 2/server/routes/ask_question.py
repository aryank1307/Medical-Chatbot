from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pinecone import Pinecone
from typing import List
from dotenv import load_dotenv
import os
import re
import json
try:
    from server.modules.llm import get_llm_chain
    from server.modules.query_handlers import query_chain
    from server.modules.embeddings import get_embeddings_model, get_embeddings_dimension
    from server.modules.disease_predictor import (
        build_probability_markdown,
        build_disease_info_markdown,
        get_disease_symptom_catalog,
    )
    from server.logger import logger
except ModuleNotFoundError:
    from modules.llm import get_llm_chain
    from modules.query_handlers import query_chain
    from modules.embeddings import get_embeddings_model, get_embeddings_dimension
    from modules.disease_predictor import (
        build_probability_markdown,
        build_disease_info_markdown,
        get_disease_symptom_catalog,
    )
    from logger import logger

load_dotenv()
router = APIRouter()
TRIAGE_HEADERS = [
    "## Possible causes",
    "## Red flags",
    "## Next steps",
    "## When to seek urgent care",
]
CRISIS_PATTERNS = [
    "i will die",
    "want to die",
    "kill myself",
    "suicide",
    "self harm",
    "end my life",
    "i dont want to live",
    "i don't want to live",
]
HIGH_RISK_CARDIO_PATTERNS = [
    "chest pain",
    "chestpain",
    "shortness of breath",
    "shortness breath",
    "breathless",
    "breathing difficulty",
    "sweating",
    "cold sweat",
]
STROKE_PATTERNS = ["face droop", "slurred speech", "weakness", "numbness", "one side"]
SEVERE_BREATHING_PATTERNS = ["wheezing", "throat swelling", "cannot breathe", "can't breathe", "blue lips"]
DEHYDRATION_PATTERNS = ["vomiting", "diarrhea", "can't keep fluids", "not peeing", "dry mouth"]
HIGH_FEVER_PATTERNS = ["high fever", "persistent fever", "fever", "chills", "stiff neck", "rash"]
ABDOMINAL_EMERGENCY_PATTERNS = ["severe abdominal pain", "right lower pain", "blood in stool", "black stool"]
PANIC_LIKE_PATTERNS = ["panic", "anxiety attack", "palpitations", "shaking", "fear", "restless"]
MEDICAL_CONTEXT_KEYWORDS = [
    "pain",
    "fever",
    "vomit",
    "nausea",
    "headache",
    "chest",
    "breath",
    "cough",
    "cold",
    "infection",
    "disease",
    "symptom",
    "medicine",
    "medication",
    "tablet",
    "dose",
    "doctor",
    "hospital",
    "urgent",
    "care",
    "rash",
    "diarrhea",
    "diarrhoea",
]
FOLLOWUP_MEDICAL_HINTS = [
    "what should i do",
    "what medicine",
    "what medication",
    "what tablet",
    "is it serious",
    "is this serious",
    "now what",
    "next step",
    "what to take",
    "can i take",
    "should i worry",
]


class SimpleRetriever(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


def _validate_env() -> tuple[str, str, str]:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    missing = []

    if not google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not groq_api_key:
        missing.append("GROQ_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return google_api_key, pinecone_api_key, os.getenv(
        "PINECONE_INDEX_NAME", "medicalindex"
    )


def _list_index_names(pc: Pinecone) -> List[str]:
    listed_indexes = pc.list_indexes()
    if hasattr(listed_indexes, "names"):
        return listed_indexes.names()
    return [i["name"] for i in listed_indexes]


def _extract_index_dimension(index_description) -> int:
    if isinstance(index_description, dict):
        dimension = index_description.get("dimension")
        if dimension is not None:
            return int(dimension)

    dimension = getattr(index_description, "dimension", None)
    if dimension is not None:
        return int(dimension)

    raise RuntimeError("Unable to detect Pinecone index dimension.")


def _ensure_triage_format(answer: str) -> str:
    if not answer:
        return answer
    if all(header in answer for header in TRIAGE_HEADERS):
        return answer

    return (
        "## Possible causes\n"
        f"{answer.strip()}\n\n"
        "## Red flags\n"
        "- Worsening symptoms, severe pain, breathing trouble, confusion, persistent high fever, or dehydration.\n\n"
        "## Next steps\n"
        "- Seek evaluation from a licensed clinician.\n"
        "- Do not self-medicate based only on this chat.\n\n"
        "## When to seek urgent care\n"
        "- Go to urgent care or emergency services if severe warning signs are present."
    )


def _is_crisis_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(pattern in normalized for pattern in CRISIS_PATTERNS)


def _is_high_risk_cardio_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    has_chest = ("chest pain" in normalized) or ("chestpain" in normalized)
    has_breath = any(
        pattern in normalized
        for pattern in ["shortness of breath", "shortness breath", "breathless", "breathing difficulty"]
    )
    has_sweat = any(pattern in normalized for pattern in ["sweating", "cold sweat"])
    # Trigger on classic cluster or at least two high-risk signals.
    total_hits = sum(1 for p in HIGH_RISK_CARDIO_PATTERNS if p in normalized)
    return (has_chest and (has_breath or has_sweat)) or total_hits >= 2


def _crisis_response() -> str:
    return (
        "I am really glad you reached out. You deserve immediate support, and you do not have to handle this alone.\n\n"
        "## Possible causes\n"
        "- Intense stress, emotional overwhelm, and hopelessness can make everything feel unbearable.\n\n"
        "## Red flags\n"
        "- Thoughts of harming yourself, feeling unsafe, or feeling unable to stay in control.\n\n"
        "## Next steps\n"
        "- Contact someone you trust right now and tell them you need support.\n"
        "- Reach a mental health professional or local emergency services immediately.\n\n"
        "## When to seek urgent care\n"
        "- If you might harm yourself or feel in immediate danger, call emergency services now.\n"
        "- In the U.S. and Canada, call or text 988 (Suicide & Crisis Lifeline) right now."
    )


def _stroke_response() -> str:
    return (
        "These symptoms could indicate a stroke and need emergency care now.\n\n"
        "## Possible causes\n- Could be a neurologic emergency; this cannot be confirmed by chat.\n\n"
        "## Red flags\n- Face droop, arm weakness, speech trouble, sudden confusion, vision loss, severe sudden headache.\n\n"
        "## Next steps\n- Call emergency services now (911 in the U.S.).\n- Note symptom start time.\n\n"
        "## When to seek urgent care\n- Immediately now. Do not wait."
    )


def _severe_breathing_response() -> str:
    return (
        "Breathing-related symptoms can become dangerous quickly.\n\n"
        "## Possible causes\n- Could be severe asthma/allergy/infection or another urgent lung issue.\n\n"
        "## Red flags\n- Trouble breathing at rest, blue lips, throat swelling, chest tightness, confusion.\n\n"
        "## Next steps\n- Sit upright, avoid exertion, use prescribed rescue inhaler if available.\n- Call emergency services now if breathing is hard.\n\n"
        "## When to seek urgent care\n- Immediately if breathing is difficult or worsening."
    )


def _dehydration_response() -> str:
    return (
        "This may be dehydration or infection-related illness and needs close monitoring.\n\n"
        "## Possible causes\n- Could be viral/bacterial stomach illness or fluid loss from repeated vomiting/diarrhea.\n\n"
        "## Red flags\n- Very low urine, dizziness/fainting, confusion, blood in vomit/stool, inability to keep fluids down.\n\n"
        "## Next steps\n- Take small frequent oral fluids/electrolytes.\n- Seek same-day medical care if symptoms persist.\n\n"
        "## When to seek urgent care\n- Go urgently if red flags appear or fluids cannot be kept down."
    )


def _high_fever_response() -> str:
    return (
        "High or persistent fever can have many causes and may need in-person evaluation.\n\n"
        "## Possible causes\n- Could be infection or inflammatory illness; exact cause cannot be confirmed via chat.\n\n"
        "## Red flags\n- Fever with breathing difficulty, confusion, stiff neck, severe headache, rash, chest pain, or dehydration.\n\n"
        "## Next steps\n- Rest, hydrate, track temperature and associated symptoms.\n- Arrange same-day clinical review if fever persists.\n\n"
        "## When to seek urgent care\n- Seek urgent care now if red flags are present."
    )


def _abdominal_emergency_response() -> str:
    return (
        "Severe abdominal symptoms can be serious and should be evaluated promptly.\n\n"
        "## Possible causes\n- Could be appendicitis, gallbladder, ulcer, bowel, or other urgent causes.\n\n"
        "## Red flags\n- Severe localized pain, persistent vomiting, black/bloody stool, fainting, fever with worsening pain.\n\n"
        "## Next steps\n- Avoid heavy food, stay hydrated if possible.\n- Get urgent in-person medical evaluation.\n\n"
        "## When to seek urgent care\n- Immediately if pain is severe or worsening."
    )


def _panic_like_response() -> str:
    return (
        "What you're feeling can be very distressing, and you're not alone.\n\n"
        "## Possible causes\n"
        "- Higher possibility: anxiety/panic-related episode.\n"
        "- Moderate possibility: stress-related physical response.\n"
        "- Lower possibility: other medical causes that can mimic panic and should be checked if symptoms persist.\n\n"
        "## Red flags\n- Chest pain, fainting, breathing trouble, new neurologic symptoms, or symptoms not settling.\n\n"
        "## Next steps\n- Slow breathing: inhale 4 seconds, exhale 6 seconds for a few minutes.\n- Sip water, sit down, and talk to someone you trust.\n- If episodes keep recurring, book a mental-health/medical review.\n\n"
        "## When to seek urgent care\n- Seek urgent care if red flags are present or symptoms feel severe."
    )


def _high_risk_cardio_response() -> str:
    return (
        "I'm really glad you told me this. Chest pain with shortness of breath or sweating can be serious, and it should be checked urgently.\n\n"
        "## Possible causes\n"
        "- Moderate possibility: anxiety can worsen these sensations.\n"
        "- Higher possibility: heart or lung emergency that needs immediate exclusion.\n"
        "- It cannot be safely confirmed from chat alone.\n\n"
        "## Red flags\n"
        "- Chest pressure/pain, breathing difficulty, heavy sweating, nausea, dizziness, or pain spreading to arm/jaw/back.\n\n"
        "## Next steps\n"
        "- Stop activity, sit upright, and stay with someone if possible.\n"
        "- Call emergency services now (911 in the U.S.).\n"
        "- If symptoms resolve, still seek same-day medical evaluation.\n\n"
        "## When to seek urgent care\n"
        "- Seek emergency care immediately now. Do not delay to self-treat this at home."
    )


def _sanitize_general_response(answer: str) -> str:
    if not answer:
        return answer
    cleaned = answer
    cleaned = re.sub(
        r"(?i)the context primarily discusses [^.]*\.\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)provided documents (primarily|mostly) [^.]*\.\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\bdiabetes\b",
        "a specific condition",
        cleaned,
    )
    return cleaned


def _uncertainty_fallback(question: str, specialty: str) -> str:
    specialty_hint = specialty if specialty != "general" else "general medicine"
    return (
        "## Possible causes\n"
        f"- Higher possibility: common short-term causes seen in {specialty_hint} concerns.\n"
        f"- Moderate possibility: inflammation/infection/stress-related effects depending on associated signs.\n"
        "- Lower possibility: less common causes that still need evaluation if symptoms continue.\n"
        "- The cause can vary based on timing, severity, age, medical history, and exposures.\n"
        f"- If symptoms are new, getting worse, or not improving, the likelihood of needing in-person evaluation increases.\n\n"
        "## Red flags\n"
        "- Severe breathing difficulty, chest pain, fainting, confusion, high persistent fever, severe dehydration, uncontrolled vomiting, or severe worsening pain.\n"
        "- Any sudden neurological change (weakness, trouble speaking, severe unusual headache) needs urgent care.\n\n"
        "## Next steps\n"
        "- Track symptom pattern: when it started, what makes it better/worse, and any associated signs.\n"
        "- Stay hydrated, rest, and seek same-day clinician advice if symptoms persist or interfere with daily function.\n"
        "- I can help you prepare a focused symptom summary for a doctor visit.\n\n"
        "## When to seek urgent care\n"
        "- Seek urgent in-person care now if any red flags are present.\n"
        "- If you are unsure whether symptoms are serious, it is safer to get immediate medical evaluation."
    )


def _merge_probability_section(question: str, answer: str) -> str:
    probability_md = build_probability_markdown(question)
    if not probability_md:
        return answer
    if not answer:
        return probability_md
    return f"{probability_md}\n\n---\n\n{answer}"


def _contains_any(text: str, patterns: list[str]) -> bool:
    normalized = (text or "").strip().lower()
    return any(p in normalized for p in patterns)


def _special_symptom_response(question: str) -> str | None:
    q = (question or "").strip().lower()
    if _contains_any(q, STROKE_PATTERNS):
        return _stroke_response()
    if _contains_any(q, SEVERE_BREATHING_PATTERNS):
        return _severe_breathing_response()
    if _contains_any(q, ABDOMINAL_EMERGENCY_PATTERNS):
        return _abdominal_emergency_response()
    if _contains_any(q, DEHYDRATION_PATTERNS):
        return _dehydration_response()
    if _contains_any(q, HIGH_FEVER_PATTERNS) and (
        "persistent" in q or "3 days" in q or "4 days" in q or "5 days" in q
    ):
        return _high_fever_response()
    if _contains_any(q, PANIC_LIKE_PATTERNS):
        return _panic_like_response()
    return None


def _build_contextual_question(question: str, chat_history: str | None) -> str:
    if not chat_history:
        return question
    try:
        parsed = json.loads(chat_history)
        if not isinstance(parsed, list):
            return question
        recent = parsed[-6:]
        lines = []
        for msg in recent:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role not in {"user", "assistant"}:
                continue
            lines.append(f"{role}: {content}")
        if not lines:
            return question
        return (
            "Conversation context:\n"
            + "\n".join(lines)
            + f"\n\nCurrent user question:\n{question}"
        )
    except Exception:
        return question


def _is_medical_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in MEDICAL_CONTEXT_KEYWORDS)


def _should_use_history(question: str) -> bool:
    normalized = (question or "").strip().lower()
    if not normalized:
        return False
    if _is_medical_text(normalized):
        return True
    return any(hint in normalized for hint in FOLLOWUP_MEDICAL_HINTS)


@router.get("/disease_catalog/")
async def disease_catalog():
    try:
        return {"diseases": get_disease_symptom_catalog()}
    except Exception as e:
        logger.exception("Error loading disease catalog")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/ask/")
async def ask_question(
    question: str = Form(...),
    namespace: str = Form("medical-guidelines"),
    specialty: str = Form("general"),
    top_k: int = Form(5),
    chat_history: str = Form("[]"),
):
    try:
        clean_question = (question or "").strip()
        use_history = _should_use_history(clean_question)
        effective_question = (
            _build_contextual_question(clean_question, chat_history)
            if use_history
            else clean_question
        )
        logger.info(f"user query: {question}")

        if not _is_medical_text(clean_question):
            return {
                "response": (
                    "I am focused on medical guidance in this chat. "
                    "Please ask a health-related question (symptoms, disease info, medication safety, diet, or next steps)."
                ),
                "sources": [],
                "namespace": (namespace or "medical-guidelines").strip(),
                "specialty": (specialty or "general").strip().lower().replace(" ", "-"),
            }

        if _is_crisis_text(effective_question):
            return {
                "response": _crisis_response(),
                "sources": [],
                "namespace": (namespace or "medical-guidelines").strip(),
                "specialty": (specialty or "general").strip().lower().replace(" ", "-"),
            }
        if _is_high_risk_cardio_text(effective_question):
            return {
                "response": _high_risk_cardio_response(),
                "sources": [],
                "namespace": (namespace or "medical-guidelines").strip(),
                "specialty": (specialty or "general").strip().lower().replace(" ", "-"),
            }
        disease_info_response = build_disease_info_markdown(effective_question)
        if disease_info_response:
            return {
                "response": disease_info_response,
                "sources": [],
                "namespace": (namespace or "medical-guidelines").strip(),
                "specialty": (specialty or "general").strip().lower().replace(" ", "-"),
            }
        symptom_response = _special_symptom_response(effective_question)
        if symptom_response:
            symptom_response = _merge_probability_section(effective_question, symptom_response)
            return {
                "response": symptom_response,
                "sources": [],
                "namespace": (namespace or "medical-guidelines").strip(),
                "specialty": (specialty or "general").strip().lower().replace(" ", "-"),
            }

        # Embed model + Pinecone setup
        google_api_key, pinecone_api_key, pinecone_index_name = _validate_env()
        os.environ["GOOGLE_API_KEY"] = google_api_key

        pc = Pinecone(api_key=pinecone_api_key)
        index_names = _list_index_names(pc)
        if pinecone_index_name not in index_names:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Pinecone index '{pinecone_index_name}' was not found. "
                        "Upload at least one PDF first, or set "
                        "PINECONE_INDEX_NAME to your existing index name."
                    )
                },
            )
        index = pc.Index(pinecone_index_name)
        embed_model = get_embeddings_model()
        embedding_dimension = get_embeddings_dimension()
        index_dimension = _extract_index_dimension(pc.describe_index(pinecone_index_name))
        if embedding_dimension != index_dimension:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Pinecone index '{pinecone_index_name}' dimension is "
                        f"{index_dimension}, but embedding model dimension is "
                        f"{embedding_dimension}. Use a new PINECONE_INDEX_NAME in "
                        "server/.env and re-upload your PDFs."
                    )
                },
            )
        embedded_query = embed_model.embed_query(effective_question)
        clean_namespace = (namespace or "medical-guidelines").strip()
        clean_specialty = (specialty or "general").strip().lower().replace(" ", "-")
        query_kwargs = {
            "vector": embedded_query,
            "top_k": max(1, min(top_k, 10)),
            "include_metadata": True,
            "namespace": clean_namespace,
        }
        if clean_specialty != "general":
            query_kwargs["filter"] = {"specialty": {"$eq": clean_specialty}}

        res = index.query(**query_kwargs)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        if not docs:
            return {
                "response": _merge_probability_section(
                    effective_question, _uncertainty_fallback(effective_question, clean_specialty)
                ),
                "sources": [],
                "namespace": clean_namespace,
                "specialty": clean_specialty,
            }

        retriever = SimpleRetriever(docs=docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, effective_question)
        if isinstance(result, dict):
            response = _sanitize_general_response(result.get("response", ""))
            result["response"] = _ensure_triage_format(response)
            result["response"] = _merge_probability_section(effective_question, result["response"])

        logger.info("query successful")
        result["namespace"] = clean_namespace
        result["specialty"] = clean_specialty
        return result

    except RuntimeError as e:
        logger.warning(str(e))
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
