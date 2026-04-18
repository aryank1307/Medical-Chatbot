import streamlit as st
import requests
from utils.api import upload_pdfs_api

NAMESPACE_OPTIONS = [
    "medical-guidelines",
    "medical-research",
    "medical-eval",
    "custom",
]

SPECIALTY_OPTIONS = [
    "general",
    "cardiology",
    "respiratory",
    "endocrine",
    "infectious",
    "neurology",
    "gastroenterology",
    "renal",
    "oncology",
    "mental-health",
    "pediatrics",
    "women-health",
    "emergency",
    "custom",
]


def render_uploader():
    st.sidebar.header("Upload Medical documents (.PDFs)")
    namespace_choice = st.sidebar.selectbox("Namespace", NAMESPACE_OPTIONS, index=0)
    namespace = (
        st.sidebar.text_input("Custom namespace", value="medical-guidelines").strip()
        if namespace_choice == "custom"
        else namespace_choice
    )
    specialty_choice = st.sidebar.selectbox("Specialty tag", SPECIALTY_OPTIONS, index=0)
    specialty = (
        st.sidebar.text_input("Custom specialty", value="general").strip()
        if specialty_choice == "custom"
        else specialty_choice
    )
    uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Upload DB") and uploaded_files:
        try:
            response = upload_pdfs_api(uploaded_files, specialty=specialty, namespace=namespace)
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(
                "Cannot connect to backend at http://127.0.0.1:8000. "
                "Start backend first using: cd server && python3 -m uvicorn main:app --host 127.0.0.1 --port 8000"
            )
            st.sidebar.caption(f"Details: {exc}")
            return

        if response.status_code == 200:
            st.sidebar.success("Uploaded successfully")
        else:
            st.sidebar.error(f"Error:{response.text}")
