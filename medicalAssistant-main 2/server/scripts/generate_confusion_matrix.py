from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from server.routes.ask_question import (
        _contains_any,
        _is_crisis_text,
        _is_high_risk_cardio_text,
        ABDOMINAL_EMERGENCY_PATTERNS,
        DEHYDRATION_PATTERNS,
        HIGH_FEVER_PATTERNS,
        PANIC_LIKE_PATTERNS,
        SEVERE_BREATHING_PATTERNS,
        STROKE_PATTERNS,
    )
except ModuleNotFoundError:
    from routes.ask_question import (  # type: ignore
        _contains_any,
        _is_crisis_text,
        _is_high_risk_cardio_text,
        ABDOMINAL_EMERGENCY_PATTERNS,
        DEHYDRATION_PATTERNS,
        HIGH_FEVER_PATTERNS,
        PANIC_LIKE_PATTERNS,
        SEVERE_BREATHING_PATTERNS,
        STROKE_PATTERNS,
    )


CLASSES = [
    "crisis",
    "high_risk_cardio",
    "stroke",
    "severe_breathing",
    "abdominal_emergency",
    "dehydration",
    "high_fever",
    "panic_like",
    "general",
]


def classify_question(question: str) -> str:
    q = (question or "").strip().lower()
    if _is_crisis_text(q):
        return "crisis"
    if _is_high_risk_cardio_text(q):
        return "high_risk_cardio"
    if _contains_any(q, STROKE_PATTERNS):
        return "stroke"
    if _contains_any(q, SEVERE_BREATHING_PATTERNS):
        return "severe_breathing"
    if _contains_any(q, ABDOMINAL_EMERGENCY_PATTERNS):
        return "abdominal_emergency"
    if _contains_any(q, DEHYDRATION_PATTERNS):
        return "dehydration"
    if _contains_any(q, HIGH_FEVER_PATTERNS) and (
        "persistent" in q or "3 days" in q or "4 days" in q or "5 days" in q
    ):
        return "high_fever"
    if _contains_any(q, PANIC_LIKE_PATTERNS):
        return "panic_like"
    return "general"


EVAL_SET: list[tuple[str, str]] = [
    ("crisis", "I want to die and cannot handle this anymore."),
    ("crisis", "I don't want to live."),
    ("crisis", "I might kill myself tonight."),
    ("crisis", "I feel like ending my life."),
    ("high_risk_cardio", "I have chest pain and shortness of breath."),
    ("high_risk_cardio", "Chestpain with cold sweat and nausea."),
    ("high_risk_cardio", "My chest pain and sweating got worse."),
    ("high_risk_cardio", "Breathless with chest pain after walking."),
    ("stroke", "My face droop started and speech is slurred."),
    ("stroke", "Sudden one side weakness and numbness in arm."),
    ("stroke", "Weakness and trouble speaking suddenly happened."),
    ("stroke", "Numbness on one side with slurred speech."),
    ("severe_breathing", "I cannot breathe and my throat feels swelling."),
    ("severe_breathing", "Severe wheezing and breathing difficulty at rest."),
    ("severe_breathing", "Blue lips and chest tightness right now."),
    ("severe_breathing", "Can't breathe properly and getting breathless."),
    ("abdominal_emergency", "Severe abdominal pain with black stool."),
    ("abdominal_emergency", "Right lower pain and vomiting since morning."),
    ("abdominal_emergency", "Blood in stool with severe abdomen cramps."),
    ("abdominal_emergency", "Severe abdominal pain is worsening quickly."),
    ("dehydration", "Vomiting and diarrhea, not peeing much."),
    ("dehydration", "Dry mouth and can't keep fluids down."),
    ("dehydration", "Diarrhea since yesterday with dizziness."),
    ("dehydration", "Repeated vomiting and not peeing."),
    ("high_fever", "Persistent fever for 4 days with chills."),
    ("high_fever", "High fever and rash for 5 days."),
    ("high_fever", "Fever with stiff neck for 3 days."),
    ("high_fever", "Persistent high fever and severe headache."),
    ("panic_like", "I think this is a panic attack with shaking."),
    ("panic_like", "Palpitations and fear and restlessness suddenly."),
    ("panic_like", "Anxiety attack happened while in crowd."),
    ("panic_like", "Panic with trembling and fear at night."),
    ("general", "What foods help with mild iron deficiency?"),
    ("general", "How much water should an adult drink daily?"),
    ("general", "Tips for better sleep hygiene and routine?"),
    ("general", "What are side effects of vitamin D supplements?"),
]


def build_confusion_matrix(
    rows: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, int]], int]:
    matrix: dict[str, dict[str, int]] = {
        t: {p: 0 for p in CLASSES} for t in CLASSES
    }
    correct = 0
    for true_label, question in rows:
        pred_label = classify_question(question)
        matrix[true_label][pred_label] += 1
        if true_label == pred_label:
            correct += 1
    return matrix, correct


def class_metrics(
    matrix: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    for cls in CLASSES:
        tp = matrix[cls][cls]
        fp = sum(matrix[t][cls] for t in CLASSES if t != cls)
        fn = sum(matrix[cls][p] for p in CLASSES if p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        support = sum(matrix[cls][p] for p in CLASSES)
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return metrics


def write_confusion_csv(
    output_csv: Path, matrix: dict[str, dict[str, int]]
) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASSES)
        for true_cls in CLASSES:
            writer.writerow([true_cls] + [matrix[true_cls][pred] for pred in CLASSES])


def write_report(
    output_txt: Path,
    matrix: dict[str, dict[str, int]],
    total: int,
    correct: int,
    metrics: dict[str, dict[str, float]],
) -> None:
    acc = correct / total if total else 0.0
    lines = [
        "Triage Routing Evaluation Report",
        f"Total samples: {total}",
        f"Correct predictions: {correct}",
        f"Accuracy: {acc:.4f}",
        "",
        "Per-class metrics:",
        "class,precision,recall,f1,support",
    ]
    for cls in CLASSES:
        m = metrics[cls]
        lines.append(
            f"{cls},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f},{int(m['support'])}"
        )

    lines.extend(["", "Confusion matrix (rows=true, cols=pred):"])
    lines.append("true\\pred," + ",".join(CLASSES))
    for true_cls in CLASSES:
        row = ",".join(str(matrix[true_cls][pred]) for pred in CLASSES)
        lines.append(f"{true_cls},{row}")

    output_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix, correct = build_confusion_matrix(EVAL_SET)
    metrics = class_metrics(matrix)
    total = len(EVAL_SET)

    csv_path = out_dir / "confusion_matrix.csv"
    txt_path = out_dir / "classification_report.txt"

    write_confusion_csv(csv_path, matrix)
    write_report(txt_path, matrix, total, correct, metrics)

    print(f"Saved confusion matrix to: {csv_path}")
    print(f"Saved classification report to: {txt_path}")


if __name__ == "__main__":
    main()
