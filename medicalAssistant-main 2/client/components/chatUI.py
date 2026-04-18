import requests
import streamlit as st

from utils.api import ask_question, get_disease_catalog

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


def render_chat():
    st.subheader("Chat with your assistant")
    namespace_choice = st.sidebar.selectbox("Query namespace", NAMESPACE_OPTIONS, index=0)
    namespace = (
        st.sidebar.text_input("Custom query namespace", value="medical-guidelines").strip()
        if namespace_choice == "custom"
        else namespace_choice
    )
    specialty_choice = st.sidebar.selectbox("Query specialty", SPECIALTY_OPTIONS, index=0)
    specialty = (
        st.sidebar.text_input("Custom query specialty", value="general").strip()
        if specialty_choice == "custom"
        else specialty_choice
    )
    with st.sidebar.expander("Disease Symptoms Catalog", expanded=False):
        try:
            catalog_response = get_disease_catalog()
            if catalog_response.status_code == 200:
                diseases = catalog_response.json().get("diseases", {})
                disease_names = sorted(diseases.keys())
                if disease_names:
                    selected_disease = st.selectbox(
                        "Select disease",
                        disease_names,
                        index=0,
                    )
                    st.markdown("**Symptoms**")
                    for symptom in diseases.get(selected_disease, []):
                        st.markdown(f"- {symptom}")
                else:
                    st.caption("No disease catalog available.")
            else:
                st.caption("Could not load disease catalog.")
        except requests.exceptions.RequestException:
            st.caption("Backend not reachable for disease catalog.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Type your question....")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Send recent turns so backend can handle follow-up questions with context.
            recent_history = st.session_state.messages[-8:]
            response = ask_question(
                user_input,
                specialty=specialty,
                namespace=namespace,
                top_k=5,
                chat_history=recent_history,
            )
        except requests.exceptions.RequestException as exc:
            st.error(
                "Cannot connect to backend at http://127.0.0.1:8000. "
                "Start backend first using: cd server && python3 -m uvicorn main:app --host 127.0.0.1 --port 8000"
            )
            st.caption(f"Details: {exc}")
            return

        if response.status_code == 200:
            data = response.json()
            answer = data["response"]
            st.chat_message("assistant").markdown(answer)
            st.caption(
                f"namespace: {data.get('namespace', namespace)} | "
                f"specialty: {data.get('specialty', specialty)}"
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.text}")
