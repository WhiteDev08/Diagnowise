from crewai import Task
from agents import emergency_agent

def create_firstaid_task(user_input):
    return Task(
        description=f"Provide comprehensive first aid guidance for the following situation: {user_input}. Use the first aid manual to ensure accuracy and provide step-by-step instructions.",
        expected_output="A well-structured, step-by-step first aid solution with clear instructions that can help save lives. Include any warnings, precautions, and when to seek professional medical help.",
        agent=emergency_agent,
        output_file="firstaid.md"
    )

