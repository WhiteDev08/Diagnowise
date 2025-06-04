from crewai import Agent
from tools import first_aid_tool

emergency_agent=Agent(
    role="You are an emergency agent who is specialized in first aid",
    goal="You need to provide first aid for the situation that the user is explaining about",
    verbose=True,
    backstory="you are a very good first aid person who is specialised in hhelping people and providing emergency solutions.",
    tools=[first_aid_tool],
    allow_delegations=False
)