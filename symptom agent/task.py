from crewai import Task

def create_diagnosis_task(agent, user_symptoms, matched_diseases):
    symptom_str = ", ".join(user_symptoms)
    if not matched_diseases:
        description = (
            f"Patient reports symptoms: {symptom_str}. "
            "No close matches were found in the knowledge graph. "
            "Suggest general possible causes and when they should see a doctor."
        )
    else:
        disease_list = [
            f"{d['disease']} (matched symptoms: {', '.join(d['matched_symptoms'])})"
            for d in matched_diseases
        ]
        diseases_str = "; ".join(disease_list)
        description = (
            f"The patient reports these symptoms: {symptom_str}.\n"
            f"The top possible matching diseases are: {diseases_str}.\n"
            "For each disease, explain your reasoning for the match and suggest next steps for the patient."
        )

    task = Task(
        description=description,
        agent=agent,
        expected_output="A list of possible diagnoses with explanations for each, and clear, actionable next steps or advice for the user."
    )
    return task
