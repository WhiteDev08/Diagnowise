# agent.py
from crewai import Agent
from tools import extract_medical_features
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model='gpt-3.5-turbo')

medical_history_agent = Agent(
    role="Medical History Analyzer",
    goal="Analyze patient history and extract risk factors, medication alerts, and a summary.",
    verbose=True,
    backstory="An expert at parsing and summarizing complex patient medical histories for healthcare providers.",
    tools=[extract_medical_features],
    llm=llm
)
