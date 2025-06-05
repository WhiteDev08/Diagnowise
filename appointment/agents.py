from crewai import Agent
from langchain_openai import ChatOpenAI
from .tools import parse_user_input, determine_routing_strategy
import os

# Import from dedicated agent modules
from EmergencyAgent.agents import emergency_agent
from HistoryAgent.agents import medical_history_agent  
from SymptomAgent.agents import create_symptom_checker_agent

class HealthcareAgents:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.3)
    
    def routing_agent(self) -> Agent:
        """Smart routing agent that parses natural language input and routes to appropriate agents"""
        return Agent(
            role="Intelligent Healthcare Router",
            goal="Parse natural language descriptions from patients and intelligently route to appropriate medical agents based on urgency, symptoms, and intent",
            backstory="""You are an advanced AI routing system that understands natural human language. 
                        When patients describe their health concerns in everyday language, you analyze the text to:
                        1. Extract key medical symptoms and concerns
                        2. Determine urgency level (Emergency/Urgent/Routine)
                        3. Identify the most appropriate medical specialist
                        4. Route to the correct combination of agents for comprehensive care
                        You understand context, implied meanings, and can detect emergency situations from descriptions.""",
            tools=[parse_user_input, determine_routing_strategy],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    def emergency_alert_agent(self) -> Agent:
        """Emergency detection and alert system - imported from EmergencyAgent module"""
        return emergency_agent()

    def create_medical_history_agent(self) -> Agent:
        """Medical history analyzer - imported from HistoryAgent module"""
        return medical_history_agent()

    def create_symptom_analyzer(self) -> Agent:
        """Clinical symptom analyzer - imported from SymptomAgent module"""
        return create_symptom_checker_agent()
    
    def create_appointment_scheduler(self) -> Agent:
        """Healthcare appointment coordinator - core appointment agent functionality"""
        return Agent(
            role="Healthcare Appointment Coordinator",
            goal="Match patients with appropriate specialists and optimize appointment scheduling",
            backstory="Intelligent scheduling system that considers symptom urgency, specialist availability, "
                     "and patient needs to coordinate optimal healthcare appointments.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_triage_agent(self) -> Agent:
        """Agent for initial patient triage and urgency assessment"""
        return Agent(
            role="Medical Triage Specialist",
            goal="Perform initial patient assessment and determine urgency level for appropriate care routing",
            backstory="Experienced triage nurse AI that quickly assesses patient conditions, determines "
                     "priority levels, and routes patients to appropriate care based on symptom severity "
                     "and medical urgency protocols.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_general_practitioner_agent(self) -> Agent:
        """General practitioner agent for routine health consultations"""
        return Agent(
            role="General Practitioner AI",
            goal="Provide general health consultations, wellness advice, and routine medical guidance",
            backstory="AI general practitioner with broad medical knowledge for routine consultations, "
                     "preventive care advice, health screenings, and general medical questions. "
                     "Refers to specialists when specialized care is needed.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )