import os
import json
from typing import Any, Dict, List, TypedDict

import requests
from langgraph.graph import END, StateGraph

from rag import retrieve_guidelines


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DISCLAIMER = (
    "This AI-generated health report is for educational purposes only and is "
    "not a medical diagnosis, treatment plan, or substitute for professional "
    "medical advice. Patients should consult a qualified healthcare provider "
    "for clinical evaluation and decisions."
)

REQUIRED_PATIENT_KEYS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "risk_probability",
    "risk_level",
]


class HealthAgentState(TypedDict, total=False):
    patient: Dict[str, Any]
    risk_factors: List[str]
    guideline_query: str
    guidelines: List[str]
    draft_report: Dict[str, Any]
    report: Dict[str, Any]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_safe_patient(patient: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in patient.items()}


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _risk_probability_pct(patient: Dict[str, Any]) -> str:
    probability = _to_float(patient.get("risk_probability"))
    if probability <= 1:
        probability *= 100
    return f"{probability:.0f}%"


def _clean_risk_level(value: Any) -> str:
    risk_level = str(value or "Unknown").replace("Risk", "").strip()
    if "High" in risk_level:
        return "🔴 High Risk"
    if "Medium" in risk_level:
        return "🟡 Medium Risk"
    if "Low" in risk_level:
        return "🟢 Low Risk"
    return risk_level


def _extract_sources(guidelines: List[str]) -> List[str]:
    sources = []
    for guideline in guidelines:
        source = guideline.split(":", 1)[0].strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _validate_patient(patient: Dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_PATIENT_KEYS if key not in patient]
    if missing_keys:
        raise ValueError(f"Patient data is missing required keys: {missing_keys}")


def analyze_patient_risk(state: HealthAgentState) -> HealthAgentState:
    """
    Node 1: Analyze the structured Pima-schema patient row and identify risk factors.
    """

    patient = state["patient"]
    _validate_patient(patient)

    risk_factors = []
    glucose = _to_float(patient.get("Glucose"))
    bmi = _to_float(patient.get("BMI"))
    blood_pressure = _to_float(patient.get("BloodPressure"))
    age = _to_float(patient.get("Age"))
    pedigree = _to_float(patient.get("DiabetesPedigreeFunction"))
    insulin = _to_float(patient.get("Insulin"))

    if glucose >= 126:
        risk_factors.append(f"Elevated glucose ({glucose:.0f})")
    elif glucose >= 100:
        risk_factors.append(f"Borderline glucose ({glucose:.0f})")

    if bmi >= 30:
        risk_factors.append(f"High BMI ({bmi:.1f})")
    elif bmi >= 25:
        risk_factors.append(f"Elevated BMI ({bmi:.1f})")

    if blood_pressure >= 80:
        risk_factors.append(f"Elevated blood pressure ({blood_pressure:.0f})")

    if age >= 45:
        risk_factors.append(f"Higher age-related risk ({age:.0f} years)")

    if pedigree >= 0.8:
        risk_factors.append(f"High diabetes pedigree function ({pedigree:.2f})")

    if insulin > 200:
        risk_factors.append(f"Elevated insulin ({insulin:.0f})")

    if not risk_factors:
        risk_factors.append("No dominant individual risk factor identified")

    guideline_query = (
        f"{patient.get('risk_level')} diabetes risk patient with glucose {glucose}, "
        f"BMI {bmi}, blood pressure {blood_pressure}, age {age}. "
        f"Risk factors: {', '.join(risk_factors)}. Need lifestyle, medication, "
        "screening, and follow-up guidance."
    )

    return {
        **state,
        "risk_factors": risk_factors[:3],
        "guideline_query": guideline_query,
    }


def retrieve_medical_guidelines(state: HealthAgentState) -> HealthAgentState:
    """
    Node 2: Retrieve relevant hardcoded diabetes guideline chunks from rag.py.
    """

    guidelines = retrieve_guidelines(state["guideline_query"], top_k=3)
    return {**state, "guidelines": guidelines}


def generate_ai_report(state: HealthAgentState) -> HealthAgentState:
    """
    Node 3: Ask Groq to generate a structured patient health report.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not configured. Set it before running the health agent."
        )

    patient = _json_safe_patient(state["patient"])
    guidelines = state["guidelines"]
    risk_factors = state["risk_factors"]

    system_prompt = (
        "You are a cautious healthcare AI assistant for an educational diabetes "
        "risk assessment demo. Use only the provided structured patient values "
        "and guideline excerpts. Do not diagnose. Do not prescribe medication. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Create a concise AI health report for this patient.

Patient values:
{json.dumps(patient, indent=2)}

Detected risk factors:
{json.dumps(risk_factors, indent=2)}

Retrieved guideline excerpts:
{json.dumps(guidelines, indent=2)}

Return exactly this JSON shape:
{{
  "risk_level": "🔴 High Risk | 🟡 Medium Risk | 🟢 Low Risk",
  "risk_probability_pct": "87%",
  "key_risk_factors": ["factor 1", "factor 2", "factor 3"],
  "risk_explanation": "plain-language clinical analysis, 3-5 sentences",
  "recommendations": {{
    "lifestyle": ["recommendation 1", "recommendation 2", "recommendation 3"],
    "medication_note": "non-prescriptive medication safety note",
    "follow_up": "recommended follow-up timing and clinical checks"
  }},
  "guideline_sources": ["source 1", "source 2", "source 3"],
  "disclaimer": "{DISCLAIMER}"
}}
"""

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=45,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    draft_report = _extract_json(content)
    return {**state, "draft_report": draft_report}


def validate_report(state: HealthAgentState) -> HealthAgentState:
    """
    Node 4: Normalize the LLM output so Streamlit Tab 3 can render it safely.
    """

    patient = state["patient"]
    draft_report = state.get("draft_report") or {}
    risk_factors = state.get("risk_factors") or []
    guidelines = state.get("guidelines") or []

    recommendations = draft_report.get("recommendations")
    if not isinstance(recommendations, dict):
        recommendations = {}

    lifestyle = _ensure_list(recommendations.get("lifestyle"))
    if not lifestyle:
        lifestyle = [
            "Discuss nutrition and physical activity goals with a qualified healthcare professional.",
            "Reduce sugar-sweetened beverages and refined carbohydrates where appropriate.",
            "Aim for regular, medically safe physical activity.",
        ]

    key_risk_factors = _ensure_list(draft_report.get("key_risk_factors")) or risk_factors[:3]
    guideline_sources = _ensure_list(draft_report.get("guideline_sources")) or _extract_sources(guidelines)

    report = {
        "risk_level": draft_report.get("risk_level") or _clean_risk_level(patient.get("risk_level")),
        "risk_probability_pct": draft_report.get("risk_probability_pct") or _risk_probability_pct(patient),
        "key_risk_factors": key_risk_factors,
        "risk_explanation": draft_report.get("risk_explanation")
        or "The model output indicates diabetes risk based on the structured clinical values provided.",
        "recommendations": {
            "lifestyle": lifestyle[:5],
            "medication_note": recommendations.get("medication_note")
            or "Medication decisions should be made only by a licensed clinician after clinical evaluation.",
            "follow_up": recommendations.get("follow_up")
            or "Schedule follow-up with a qualified healthcare professional for appropriate screening and review.",
        },
        "guideline_sources": guideline_sources,
        "disclaimer": draft_report.get("disclaimer") or DISCLAIMER,
    }

    while len(report["key_risk_factors"]) < 3:
        report["key_risk_factors"].append("Additional clinical review recommended")

    return {**state, "report": report}


def build_health_agent():
    graph = StateGraph(HealthAgentState)

    graph.add_node("analyze_patient_risk", analyze_patient_risk)
    graph.add_node("retrieve_medical_guidelines", retrieve_medical_guidelines)
    graph.add_node("generate_ai_report", generate_ai_report)
    graph.add_node("validate_report", validate_report)

    graph.set_entry_point("analyze_patient_risk")
    graph.add_edge("analyze_patient_risk", "retrieve_medical_guidelines")
    graph.add_edge("retrieve_medical_guidelines", "generate_ai_report")
    graph.add_edge("generate_ai_report", "validate_report")
    graph.add_edge("validate_report", END)

    return graph.compile()


health_agent = build_health_agent()


def run_health_agent(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the 4-node LangGraph health agent for one patient.
    Called by Streamlit Tab 3.
    """

    result = health_agent.invoke({"patient": _json_safe_patient(patient_dict)})
    return result["report"]


if __name__ == "__main__":
    sample_patient = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 84,
        "SkinThickness": 35,
        "Insulin": 180,
        "BMI": 34.2,
        "DiabetesPedigreeFunction": 0.72,
        "Age": 49,
        "risk_probability": 0.82,
        "risk_level": "🔴 High Risk",
    }

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Export it first, then run: python src/agent.py"
        )

    print(json.dumps(run_health_agent(sample_patient), indent=2))
