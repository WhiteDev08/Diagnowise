# tools.py
from langchain_openai import ChatOpenAI
from crewai.tools import tool
from langchain.schema import HumanMessage
import json
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

@tool
def extract_medical_features(medical_history: str) -> dict:
    """Uses LangChain LLM to extract risk factors, medication alerts, and summary."""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.3, max_tokens=400)

    prompt = (
        "You are a medical data analyst. Extract the following from the patient's medical history:\n"
        "- List of key risk factors (diseases, family history, lifestyle risks).\n"
        "- Any medication alerts (interactions, allergies, dangerous drugs).\n"
        "- Summarize the key health events.\n"
        "Return JSON in this format:\n"
        "{"
        '"risk_factors": [...],'
        '"medication_alerts": [...],'
        '"summary": "..."'
        "}\n\n"
        f"Patient medical history:\n{medical_history}\n"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    output_text = response.content
    try:
        result = json.loads(output_text)
        return json.dumps(result, indent=2)
    except Exception:
        result = {"error": "Could not parse model output.", "raw_output": output_text}
        return json.dumps(result, indent=2)