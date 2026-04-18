import requests
from config import API_URL
import json

UPLOAD_TIMEOUT_SECONDS = 300
ASK_TIMEOUT_SECONDS = 120


def upload_pdfs_api(files, specialty="general", namespace="medical-guidelines"):
    files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in files]
    return requests.post(
        f"{API_URL}/upload_pdfs/",
        files=files_payload,
        data={"specialty": specialty, "namespace": namespace},
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )

def ask_question(
    question,
    specialty="general",
    namespace="medical-guidelines",
    top_k=5,
    chat_history=None,
):
    return requests.post(
        f"{API_URL}/ask/",
        data={
            "question": question,
            "specialty": specialty,
            "namespace": namespace,
            "top_k": top_k,
            "chat_history": json.dumps(chat_history or []),
        },
        timeout=ASK_TIMEOUT_SECONDS,
    )


def get_disease_catalog():
    return requests.get(f"{API_URL}/disease_catalog/", timeout=ASK_TIMEOUT_SECONDS)
