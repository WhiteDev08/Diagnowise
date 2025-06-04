from crewai import Task
from agents import medical_history_agent


def create_history_analysis_task(medical_history: str, agent):
    """Creates a CrewAI task to analyze given medical history."""
    return Task(
        description=(
            "Analyze this patient's medical history:\n"
            f"{medical_history}\n"
        ),
        expected_output="Return JSON with 'risk_factors', 'medication_alerts', and 'summary'.",
        agent=agent
    )
