from crewai import Agent
from langchain_openai import ChatOpenAI

def create_symptom_checker_agent(openai_api_key):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )
    agent = Agent(
        role="Medical Symptom Checker",
        goal="Suggest possible diseases for a given set of symptoms, explain reasoning, and recommend next steps.",
        backstory=(
            "You are a highly experienced medical AI. "
            "Given a set of symptoms, you match them to known diseases and explain to the user possible diagnoses and what actions to take next."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    return agent
