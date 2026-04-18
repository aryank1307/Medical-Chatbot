import csv
import difflib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "disease_nb_model.json"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "datasets" / "Training.csv"
MIN_SYMPTOMS_FOR_RANKING = 4
HIGH_RISK_MIN_SYMPTOMS = 6
HIGH_RISK_DISEASES = {"Dengue", "Malaria", "AIDS", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "hepatitis A", "Tuberculosis"}

_MODEL_CACHE: Optional[dict] = None

COMMON_SYMPTOM_SYNONYMS: Dict[str, List[str]] = {
    "headache": ["head pain", "head hurts", "migraine", "heavy head"],
    "high fever": ["high temperature", "very high fever", "feeling hot", "hot body"],
    "mild fever": ["low fever", "low grade fever", "slight fever"],
    "nausea": ["queasy", "vomit feeling", "feel like vomiting", "sick to my stomach"],
    "vomiting": ["throwing up", "throw up", "puking", "vomit"],
    "diarrhoea": ["diarrhea", "loose motion", "loose stools", "watery stool"],
    "abdominal pain": ["stomach ache", "stomach pain", "belly pain", "tummy pain"],
    "joint pain": ["body ache", "body pain", "aching joints"],
    "breathlessness": ["shortness of breath", "breathless", "can't breathe well"],
    "chest pain": ["chest tightness", "pain in chest", "chest discomfort"],
    "fatigue": ["tiredness", "very tired", "low energy", "weakness"],
    "cough": ["coughing"],
    "runny nose": ["running nose", "nasal discharge"],
    "congestion": ["blocked nose", "stuffy nose"],
    "sore throat": ["throat pain", "painful throat"],
    "dizziness": ["lightheaded", "giddiness", "feeling dizzy"],
}

DISEASE_HALLMARK_SYMPTOMS: Dict[str, Set[str]] = {
    "AIDS": {"extra_marital_contacts", "muscle_wasting", "patches_in_throat"},
    "Paralysis (brain hemorrhage)": {"weakness_of_one_body_side", "slurred_speech"},
    "Heart attack": {"chest_pain", "sweating", "breathlessness"},
    "Hypertension": {"chest_pain", "dizziness", "headache"},
}

GENERIC_DISEASE_ADVICE = {
    "next_steps": "Rest, hydrate, monitor temperature/symptoms, and seek same-day clinician review if not improving.",
    "medication": "For mild fever/pain only, consider OTC paracetamol/acetaminophen as per package label if safe for you. Avoid self-starting antibiotics.",
    "diet": "Light fluids and food: ORS, soups, coconut water, khichdi, toast, banana, curd/yogurt.",
    "escalation": "Urgent care if persistent high fever, confusion, breathing trouble, severe weakness, repeated vomiting, dehydration, chest pain, or worsening condition.",
}

DISEASE_SPECIFIC_ADVICE: Dict[str, Dict[str, str]] = {
    "Malaria": {
        "next_steps": "Get urgent blood test confirmation (malaria smear/rapid test) and same-day doctor evaluation.",
        "medication": "Do not self-medicate antimalarials without prescription. For fever, OTC paracetamol/acetaminophen may be used as per label if safe.",
        "diet": "Hydration-first: ORS, water, clear soups, coconut water; soft meals.",
        "escalation": "Emergency care for confusion, persistent vomiting, breathing difficulty, jaundice, reduced urine, or very high fever.",
    },
    "Typhoid": {
        "next_steps": "Seek clinician review for testing and treatment plan; maintain strict hydration and hygiene.",
        "medication": "Avoid self-starting antibiotics. Use OTC fever relief (paracetamol/acetaminophen) as per label if safe.",
        "diet": "Soft, easy-to-digest meals; avoid oily/spicy foods; continue fluids.",
        "escalation": "Urgent care for persistent high fever, severe abdominal pain, repeated vomiting, or dehydration.",
    },
    "Dengue": {
        "next_steps": "Get same-day medical review and CBC/platelet monitoring as advised.",
        "medication": "Prefer paracetamol/acetaminophen for fever as per label if safe. Avoid aspirin/ibuprofen unless your doctor says otherwise.",
        "diet": "High fluid intake: ORS, water, soups, coconut water; light meals.",
        "escalation": "Emergency care for bleeding, severe abdominal pain, persistent vomiting, fainting, drowsiness, or reduced urine.",
    },
    "Gastroenteritis": {
        "next_steps": "Focus on oral rehydration and rest; monitor urine output and frequency of vomiting/diarrhea.",
        "medication": "Use OTC fever relief as per label if safe. Avoid unnecessary antibiotics.",
        "diet": "ORS, clear fluids, banana, rice, toast, curd/yogurt; avoid heavy/oily food.",
        "escalation": "Urgent care if unable to keep fluids down, blood in stool/vomit, dizziness, or low urine.",
    },
    "Common Cold": {
        "next_steps": "Rest, warm fluids, steam inhalation, and symptom monitoring.",
        "medication": "Paracetamol/acetaminophen for fever/body pain as per label if safe; saline nasal spray for congestion.",
        "diet": "Warm soups, fluids, light meals, fruits rich in vitamin C.",
        "escalation": "Seek care for breathlessness, high fever >3 days, chest pain, or worsening cough.",
    },
    "Pneumonia": {
        "next_steps": "Get urgent clinical evaluation and chest assessment.",
        "medication": "Do not self-start antibiotics; use only prescribed treatment. Fever relief per label if safe.",
        "diet": "Hydrate, soft nutritious meals, avoid smoking/alcohol.",
        "escalation": "Emergency care for low oxygen signs, severe breathlessness, confusion, or persistent high fever.",
    },
    "Bronchial Asthma": {
        "next_steps": "Use prescribed inhaler plan and avoid known triggers (dust/smoke/cold air).",
        "medication": "Use rescue inhaler if already prescribed; do not overuse OTC cough syrups.",
        "diet": "Hydration and balanced diet; avoid trigger foods if known.",
        "escalation": "Urgent care for severe breathlessness, wheeze at rest, or inability to speak full sentences.",
    },
    "Heart attack": {
        "next_steps": "Call emergency services immediately and stop all exertion.",
        "medication": "Do not self-manage at home; emergency medical treatment is required.",
        "diet": "No oral intake if severe symptoms are ongoing and emergency care is in progress.",
        "escalation": "Immediate emergency care now for chest pain, sweating, breathlessness, dizziness, or radiation to arm/jaw.",
    },
    "Migraine": {
        "next_steps": "Rest in a dark/quiet room, hydrate, and identify triggers.",
        "medication": "Use OTC analgesic only as per label if safe and not overused.",
        "diet": "Regular meals, hydration, limit caffeine/alcohol and trigger foods.",
        "escalation": "Seek urgent care for new severe worst-ever headache, weakness, speech issues, or persistent vomiting.",
    },
    "Diabetes ": {
        "next_steps": "Check blood glucose and schedule physician review for medication/lifestyle plan.",
        "medication": "Continue prescribed diabetes medicines; avoid dose changes without clinician advice.",
        "diet": "Low refined sugar, high-fiber balanced meals, controlled portions.",
        "escalation": "Urgent care for very high sugars, dehydration, confusion, vomiting, or drowsiness.",
    },
    "Hypoglycemia": {
        "next_steps": "Take fast-acting glucose immediately and recheck symptoms.",
        "medication": "Adjust anti-diabetic medicines only with clinician guidance.",
        "diet": "Small frequent meals; carry glucose source/snacks.",
        "escalation": "Emergency care for fainting, seizure, persistent confusion, or no response to oral glucose.",
    },
    "Hypertension ": {
        "next_steps": "Monitor BP regularly and review with clinician for long-term control.",
        "medication": "Continue prescribed BP medicines; avoid abrupt stopping.",
        "diet": "Low-salt diet, weight control, limit processed foods and alcohol.",
        "escalation": "Urgent care for severe headache, chest pain, breathlessness, neuro symptoms, or very high BP readings.",
    },
    "Tuberculosis": {
        "next_steps": "Get sputum/chest evaluation and start supervised treatment if confirmed.",
        "medication": "TB treatment must be prescribed and completed fully; do not self-medicate.",
        "diet": "High-protein nutritious diet, hydration, and rest.",
        "escalation": "Urgent care for breathlessness, coughing blood, severe weakness, or persistent high fever.",
    },
    "Urinary tract infection": {
        "next_steps": "Get urine test and clinician review.",
        "medication": "Avoid self-starting antibiotics; use prescribed antibiotics only after evaluation.",
        "diet": "Hydrate well; avoid bladder irritants (excess caffeine).",
        "escalation": "Urgent care for fever with flank pain, vomiting, confusion, or reduced urine.",
    },
    "Chicken pox": {
        "next_steps": "Isolate, monitor rash/fever, and seek clinician advice.",
        "medication": "Use fever medicine per label if safe; avoid aspirin.",
        "diet": "Hydration and soft bland food; maintain skin hygiene.",
        "escalation": "Urgent care for breathing difficulty, confusion, severe dehydration, or worsening rash.",
    },
    "Jaundice": {
        "next_steps": "Get liver function testing and clinician review promptly.",
        "medication": "Avoid alcohol and avoid non-prescribed hepatotoxic drugs.",
        "diet": "Hydrating and light low-fat meals.",
        "escalation": "Urgent care for confusion, bleeding, persistent vomiting, severe weakness, or reduced urine.",
    },
    "Hepatitis E": {
        "next_steps": "Seek clinician review with liver tests; monitor hydration and urine/stool changes.",
        "medication": "Avoid alcohol and self-medication; use only doctor-approved medicines.",
        "diet": "Light low-fat diet, fluids, and rest.",
        "escalation": "Urgent care for jaundice worsening, confusion, persistent vomiting, bleeding, or severe weakness.",
    },
    "hepatitis A": {
        "next_steps": "Get medical review with liver tests and rest.",
        "medication": "Avoid unnecessary medicines and alcohol; fever relief per label if safe.",
        "diet": "Light low-fat diet with good hydration.",
        "escalation": "Urgent care for persistent vomiting, dehydration, confusion, or bleeding signs.",
    },
    "Hepatitis B": {
        "next_steps": "Consult physician/hepatologist for evaluation and monitoring plan.",
        "medication": "Take only prescribed therapy; avoid alcohol and liver-toxic self-medication.",
        "diet": "Balanced low-fat diet and adequate hydration.",
        "escalation": "Urgent care for jaundice worsening, bleeding, confusion, or abdominal swelling.",
    },
    "Hepatitis C": {
        "next_steps": "Get specialist evaluation for confirmatory testing and antiviral plan.",
        "medication": "Do not self-treat; follow prescribed antiviral management only.",
        "diet": "Liver-friendly diet and avoid alcohol completely.",
        "escalation": "Urgent care for confusion, bleeding, severe weakness, or worsening jaundice.",
    },
    "GERD": {
        "next_steps": "Lifestyle changes and clinician review if frequent symptoms.",
        "medication": "OTC antacid may help short term; persistent symptoms need doctor-guided therapy.",
        "diet": "Small meals; avoid spicy/oily food, late-night meals, caffeine, and alcohol.",
        "escalation": "Urgent care for chest pain, vomiting blood, black stools, or trouble swallowing.",
    },
}


def _clean_token(value: str) -> str:
    cleaned = (value or "").strip().lower().replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _symptom_aliases(symptom: str) -> Set[str]:
    aliases = {_clean_token(symptom)}
    if " and " in symptom:
        aliases.add(symptom.replace(" and ", " "))
    if "feets" in symptom:
        aliases.add(symptom.replace("feets", "feet"))
    if "diarrhoea" in symptom:
        aliases.add(symptom.replace("diarrhoea", "diarrhea"))
    for phrase in COMMON_SYMPTOM_SYNONYMS.get(_clean_token(symptom), []):
        aliases.add(phrase)
    return {_clean_token(a) for a in aliases if a}


def _normalized_words(text: str) -> List[str]:
    cleaned = _clean_token(text)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return [w for w in cleaned.split() if w]


def _generate_ngrams(words: List[str], max_n: int = 4) -> Set[str]:
    ngrams: Set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(0, max(0, len(words) - n + 1)):
            ngrams.add(" ".join(words[i : i + n]))
    return ngrams


def _train_model_from_csv(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise RuntimeError("Training dataset is empty.")
        fieldnames = reader.fieldnames or []

    if "prognosis" not in fieldnames:
        raise RuntimeError("Training dataset must contain a 'prognosis' column.")

    symptoms = [c for c in fieldnames if c != "prognosis"]
    disease_counts: Dict[str, int] = {}
    symptom_pos_counts: Dict[str, Dict[str, int]] = {}
    symptom_disease_df: Dict[str, int] = {s: 0 for s in symptoms}
    total = len(rows)

    for row in rows:
        disease = (row.get("prognosis") or "").strip()
        if not disease:
            continue
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        symptom_pos_counts.setdefault(disease, {s: 0 for s in symptoms})
        for symptom in symptoms:
            val = (row.get(symptom) or "0").strip()
            try:
                present = float(val) > 0
            except ValueError:
                present = val.lower() in {"yes", "true", "1"}
            if present:
                symptom_pos_counts[disease][symptom] += 1

    if not disease_counts:
        raise RuntimeError("No prognosis labels found in dataset.")

    diseases = sorted(disease_counts.keys())
    priors = {d: disease_counts[d] / total for d in diseases}
    likelihood_pos: Dict[str, Dict[str, float]] = {}
    likelihood_neg: Dict[str, Dict[str, float]] = {}

    for d in diseases:
        n_d = disease_counts[d]
        likelihood_pos[d] = {}
        likelihood_neg[d] = {}
        for symptom in symptoms:
            pos_count = symptom_pos_counts[d][symptom]
            # Laplace smoothing for Bernoulli Naive Bayes.
            p_pos = (pos_count + 1) / (n_d + 2)
            p_neg = 1.0 - p_pos
            likelihood_pos[d][symptom] = p_pos
            likelihood_neg[d][symptom] = p_neg
        # Disease frequency for each symptom across disease classes.
        for symptom in symptoms:
            if likelihood_pos[d][symptom] >= 0.5:
                symptom_disease_df[symptom] += 1

    num_diseases = max(1, len(diseases))
    symptom_idf = {
        s: math.log((1 + num_diseases) / (1 + symptom_disease_df[s])) + 1.0
        for s in symptoms
    }

    vocabulary = [_clean_token(s) for s in symptoms]
    symptom_lookup = {
        _clean_token(s): s for s in symptoms
    }
    alias_lookup: Dict[str, str] = {}
    for canonical in symptoms:
        for alias in _symptom_aliases(_clean_token(canonical)):
            alias_lookup[alias] = canonical

    return {
        "symptoms": symptoms,
        "symptom_lookup": symptom_lookup,
        "alias_lookup": alias_lookup,
        "vocabulary": vocabulary,
        "diseases": diseases,
        "priors": priors,
        "likelihood_pos": likelihood_pos,
        "likelihood_neg": likelihood_neg,
        "symptom_idf": symptom_idf,
    }


def _persist_model(model: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(model, f)


def _load_persisted_model() -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _augment_alias_lookup(model: dict) -> dict:
    symptoms = model.get("symptoms", [])
    alias_lookup = model.get("alias_lookup", {}) or {}
    for canonical in symptoms:
        for alias in _symptom_aliases(_clean_token(canonical)):
            alias_lookup[alias] = canonical
    model["alias_lookup"] = alias_lookup
    return model


def _disease_core_symptoms(model: dict, disease: str) -> Set[str]:
    likelihood_map = model["likelihood_pos"][disease]
    core = {s for s, p in likelihood_map.items() if float(p) >= 0.6}
    if not core:
        core = {s for s, p in likelihood_map.items() if float(p) >= 0.5}
    return core


def _humanize_label(value: str) -> str:
    text = (value or "").strip().replace("_", " ")
    text = " ".join(text.split())
    return text.capitalize() if text else text


def _normalize_disease_name(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def get_or_train_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    persisted = _load_persisted_model()
    if persisted is not None:
        _MODEL_CACHE = _augment_alias_lookup(persisted)
        return _MODEL_CACHE

    dataset_path = Path(os.getenv("DISEASE_DATASET_PATH", str(DEFAULT_DATASET_PATH)))
    model = _train_model_from_csv(dataset_path)
    model = _augment_alias_lookup(model)
    _persist_model(model)
    _MODEL_CACHE = model
    return _MODEL_CACHE


def get_disease_symptom_catalog() -> Dict[str, List[str]]:
    model = get_or_train_model()
    catalog: Dict[str, List[str]] = {}
    for disease in model.get("diseases", []):
        core = sorted(_disease_core_symptoms(model, disease))
        catalog[disease] = [_humanize_label(symptom) for symptom in core]
    return catalog


def _disease_causes_line(disease: str) -> str:
    generic = "May be related to infection, inflammation, metabolic changes, or organ-specific conditions depending on history and exam."
    causes = {
        "Gastroenteritis": "Usually due to viral/bacterial gut infection, contaminated food/water, or poor food hygiene.",
        "Typhoid": "Usually due to Salmonella infection from contaminated food or water.",
        "Malaria": "Parasitic infection transmitted by mosquito bites in endemic areas.",
        "Dengue": "Viral infection spread by Aedes mosquito bites.",
        "Common Cold": "Usually viral upper respiratory infection.",
    }
    return causes.get(disease, generic)


def resolve_disease_from_text(question: str) -> Optional[str]:
    model = get_or_train_model()
    diseases = model.get("diseases", [])
    if not diseases:
        return None

    normalized_to_original = {_normalize_disease_name(d): d for d in diseases}
    q = _normalize_disease_name(question)
    if not q:
        return None

    for nd, original in normalized_to_original.items():
        if nd and nd in q:
            return original

    tokens = q.split()
    if tokens:
        for nd, original in normalized_to_original.items():
            nd_tokens = set(nd.split())
            if nd_tokens and len(nd_tokens.intersection(tokens)) >= min(2, len(nd_tokens)):
                return original

    # Fuzzy match on sentence chunks (handles misspellings inside long sentences).
    q_tokens = q.split()
    chunks = {q}
    for n in (1, 2, 3, 4):
        for i in range(0, max(0, len(q_tokens) - n + 1)):
            chunks.add(" ".join(q_tokens[i : i + n]))

    for chunk in chunks:
        close = difflib.get_close_matches(chunk, list(normalized_to_original.keys()), n=1, cutoff=0.78)
        if close:
            return normalized_to_original[close[0]]

    close = difflib.get_close_matches(q, list(normalized_to_original.keys()), n=1, cutoff=0.72)
    if close:
        return normalized_to_original[close[0]]
    return None


def build_disease_info_markdown(question: str) -> Optional[str]:
    disease = resolve_disease_from_text(question)
    if not disease:
        return None

    catalog = get_disease_symptom_catalog()
    symptoms = catalog.get(disease, [])
    advice = DISEASE_SPECIFIC_ADVICE.get(disease, GENERIC_DISEASE_ADVICE)
    symptoms_line = ", ".join(symptoms) if symptoms else "No symptom profile available."

    return "\n".join(
        [
            f"## {disease}",
            "### Possible causes",
            f"- {_disease_causes_line(disease)}",
            "### Common symptoms",
            f"- {symptoms_line}",
            "### Medication guidance",
            f"- {advice['medication']}",
            "### Diet guidance",
            f"- {advice['diet']}",
            "### Next steps",
            f"- {advice['next_steps']}",
            "### When to seek urgent care",
            f"- {advice['escalation']}",
            "",
            "This is general educational guidance, not a confirmed diagnosis.",
        ]
    )


def extract_symptoms_from_text(question: str, max_symptoms: int = 8) -> List[str]:
    model = get_or_train_model()
    text = _clean_token(question)
    words = _normalized_words(question)
    ngrams = _generate_ngrams(words, max_n=4)
    found: List[str] = []

    for phrase, canonical in model["alias_lookup"].items():
        if not phrase:
            continue
        if phrase in text or phrase in ngrams:
            found.append(canonical)

    # Mild normalization of frequent user phrasing.
    if "fever" in ngrams and "high_fever" not in found and "mild_fever" not in found:
        found.append("high_fever")
    if "head pain" in ngrams and "headache" not in found:
        found.append("headache")
    if "body ache" in ngrams and "joint_pain" not in found:
        found.append("joint_pain")

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for symptom in found:
        if symptom in seen:
            continue
        seen.add(symptom)
        deduped.append(symptom)

    return deduped[:max_symptoms]


def predict_disease_probabilities(
    symptoms_present: List[str],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    model = get_or_train_model()
    symptoms_set = set(symptoms_present)
    if not symptoms_set:
        return []

    scored: List[Tuple[str, float]] = []
    symptoms_count = max(1, len(symptoms_set))
    symptom_idf = model.get("symptom_idf", {})
    fever_present = ("high_fever" in symptoms_set) or ("mild_fever" in symptoms_set)

    for disease in model["diseases"]:
        if disease in HIGH_RISK_DISEASES and symptoms_count < HIGH_RISK_MIN_SYMPTOMS:
            continue
        disease_core = _disease_core_symptoms(model, disease)
        if not disease_core:
            continue

        matched = symptoms_set.intersection(disease_core)
        overlap = len(matched)
        if overlap == 0:
            continue

        hallmark = DISEASE_HALLMARK_SYMPTOMS.get(disease)
        if hallmark and not (symptoms_set.intersection(hallmark)):
            continue

        # Query coverage: how many user symptoms match this disease.
        query_coverage = overlap / symptoms_count
        # Specificity: how focused the disease profile is for the matched symptoms.
        specificity = overlap / len(disease_core)
        # Weighted informativeness of matched symptoms.
        info_score = sum(float(symptom_idf.get(s, 1.0)) for s in matched) / overlap

        # Strong penalty when user reports fever but disease core does not include fever.
        fever_penalty = 1.0
        if fever_present and ("high_fever" not in disease_core and "mild_fever" not in disease_core):
            fever_penalty = 0.25

        # Combined score (favor matching user symptoms, then compact disease profile, then informative symptoms).
        score = ((0.70 * query_coverage) + (0.20 * specificity) + (0.10 * min(info_score / 3.0, 1.0))) * fever_penalty
        scored.append((disease, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: max(top_n * 3, top_n)]

    total_score = sum(s for _, s in scored) or 1.0
    probs = [(d, (s / total_score) * 100.0) for d, s in scored]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[:top_n]


def build_probability_markdown(question: str) -> Optional[str]:
    try:
        detected = extract_symptoms_from_text(question)
    except Exception:
        return None
    if not detected:
        return None

    readable_detected = ", ".join(_clean_token(s) for s in detected)
    if len(detected) < MIN_SYMPTOMS_FOR_RANKING:
        return "\n".join(
            [
                "## Symptom interpretation",
                f"Detected symptoms: {readable_detected}",
                "",
                "Current input is too limited for reliable disease ranking.",
                "For now, this looks like a non-specific symptom cluster (for example fever-type illness), not a confirmed disease pattern.",
                "",
                "Please add more details such as:",
                "- symptom duration (how many days),",
                "- associated symptoms (cough, sore throat, vomiting, rash, body pain, breathlessness),",
                "- severity and red flags.",
                "",
                "## Next steps (mild symptoms only)",
                "- Rest and hydrate well (water, ORS, clear soups, coconut water).",
                "- Eat light foods: khichdi, dal-rice, toast, banana, curd/yogurt, fruits.",
                "- Avoid oily/spicy foods and alcohol until better.",
                "- For mild fever/discomfort, you may use an OTC antipyretic such as paracetamol/acetaminophen only as per the package label, if it is safe for you.",
                "- Avoid self-starting antibiotics.",
                "",
                "## When to seek medical care",
                "- Fever lasting more than 2-3 days, worsening headache, repeated vomiting, rash, breathing trouble, chest pain, confusion, fainting, or dehydration.",
                "- Seek urgent/emergency care immediately for severe symptoms or red flags.",
            ]
        )

    try:
        preds = predict_disease_probabilities(detected, top_n=5)
    except Exception:
        return None
    if not preds:
        return None

    lines = [
        "## Symptom-based disease probability (experimental)",
        f"Detected symptoms: {readable_detected}",
        "",
    ]
    for disease, pct in preds:
        lines.append(f"- {disease}: {pct:.2f}% probable")
    lines.extend(["", "## Disease-wise next steps and medication"])
    for disease, _ in preds:
        advice = DISEASE_SPECIFIC_ADVICE.get(disease, GENERIC_DISEASE_ADVICE)
        lines.extend(
            [
                f"### {disease}",
                f"- Next steps: {advice['next_steps']}",
                f"- Medication: {advice['medication']}",
                f"- Diet: {advice['diet']}",
                f"- Seek urgent care if: {advice['escalation']}",
                "",
            ]
        )
    top_pct = preds[0][1]
    if top_pct < 25:
        lines.extend(
            [
                "",
                "Confidence note: current symptom pattern is broad, so these are only rough possibilities.",
                "Add more symptoms (for example chills, rash, body pain, abdominal pain, stool changes, travel/exposure, duration in days) for better narrowing.",
            ]
        )
    lines.extend(
        [
            "",
            "This is a screening estimate from dataset patterns, not a medical diagnosis.",
        ]
    )
    return "\n".join(lines)
