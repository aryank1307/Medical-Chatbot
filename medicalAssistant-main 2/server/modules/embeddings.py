import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests

load_dotenv()

_WORKING_EMBEDDING_MODEL: Optional[str] = None
_WORKING_EMBEDDING_DIMENSION: Optional[int] = None
_DISCOVERED_EMBEDDING_MODELS: Optional[List[str]] = None


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _discover_embedding_models(google_api_key: str) -> List[str]:
    global _DISCOVERED_EMBEDDING_MODELS

    if _DISCOVERED_EMBEDDING_MODELS is not None:
        return _DISCOVERED_EMBEDDING_MODELS

    response = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": google_api_key},
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Google ListModels failed with "
            f"HTTP {response.status_code}: {response.text[:300]}"
        )

    payload = response.json()
    discovered: List[str] = []

    for model in payload.get("models", []):
        methods = [m.lower() for m in model.get("supportedGenerationMethods", [])]
        if "embedcontent" not in methods:
            continue

        name = model.get("name")
        if not name:
            continue
        discovered.append(name)
        if name.startswith("models/"):
            discovered.append(name.split("models/", 1)[1])

    _DISCOVERED_EMBEDDING_MODELS = _dedupe(discovered)
    return _DISCOVERED_EMBEDDING_MODELS


def _candidate_models(discovered_models: List[str]) -> List[str]:
    configured_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
    ordered_candidates = [
        configured_model,
        *discovered_models,
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001",
        "embedding-001",
    ]
    return _dedupe(ordered_candidates)


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    global _WORKING_EMBEDDING_MODEL, _WORKING_EMBEDDING_DIMENSION

    if _WORKING_EMBEDDING_MODEL:
        return GoogleGenerativeAIEmbeddings(model=_WORKING_EMBEDDING_MODEL)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("Missing required environment variable: GOOGLE_API_KEY")

    discovery_error: Optional[str] = None
    discovered_models: List[str] = []
    try:
        discovered_models = _discover_embedding_models(google_api_key)
    except Exception as exc:
        discovery_error = str(exc)

    candidates = _candidate_models(discovered_models)
    model_errors = []
    for model_name in candidates:
        try:
            model = GoogleGenerativeAIEmbeddings(model=model_name)
            # Probe once to ensure the selected model supports embedContent.
            probe_vector = model.embed_query("health check")
            _WORKING_EMBEDDING_MODEL = model_name
            _WORKING_EMBEDDING_DIMENSION = len(probe_vector)
            return model
        except Exception as exc:
            model_errors.append(f"{model_name}: {exc}")

    tried = ", ".join(candidates)
    discovered = ", ".join(discovered_models) if discovered_models else "none"
    list_models_note = (
        f" ListModels error: {discovery_error}." if discovery_error else ""
    )
    raise RuntimeError(
        "No supported Google embedding model is available for your API key. "
        f"Tried: {tried}. "
        f"ListModels embed-capable models: {discovered}."
        f"{list_models_note}"
        " Set GOOGLE_EMBEDDING_MODEL in server/.env to a supported model from your account. "
        f"Errors: {' | '.join(model_errors)}"
    )


def get_embeddings_dimension() -> int:
    global _WORKING_EMBEDDING_DIMENSION

    if _WORKING_EMBEDDING_DIMENSION is not None:
        return _WORKING_EMBEDDING_DIMENSION

    model = get_embeddings_model()
    _WORKING_EMBEDDING_DIMENSION = len(model.embed_query("dimension check"))
    return _WORKING_EMBEDDING_DIMENSION
