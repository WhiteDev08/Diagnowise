from crewai import Task, Agent
from .tools import HEALTHCARE_PROVIDERS
from EmergencyAgent.tasks import create_firstaid_task

from HistoryAgent.task import  create_history_analysis_task
from SymptomAgent.task import create_diagnosis_task


class HealthcareTasks:
    @staticmethod
    def create_routing_task(agent: Agent, user_description: str) -> Task:
        return Task(
            description=f"""
            Analyze this patient's natural language description and determine intelligent routing:
            
            Patient Description: "{user_description}"
            
            Your task:
            1. Parse the natural language input to extract symptoms, urgency, and medical needs
            2. Determine if this is an emergency situation requiring immediate attention
            3. Identify the most appropriate medical specialty and agents needed
            4. Create a routing strategy for optimal patient care
            
            Use the parse_user_input and determine_routing_strategy tools to analyze the input.
            
            Return structured routing decision with agent activation plan.
            """,
            agent=agent,
            expected_output="Comprehensive routing analysis with agent activation strategy and emergency assessment"
        )

    # Use EmergencyAgent
    @staticmethod
    def create_emergency_assessment_task(agent: Agent, parsed_input: str) -> Task:
        return EmergencyTasks.create_emergency_assessment_task(agent, parsed_input)

    # Use HistoryAgent
    @staticmethod
    def create_medical_history_task(agent: Agent, patient_data: dict) -> Task:
        return HistoryTasks.create_medical_history_task(agent, patient_data)

    # Use SymptomAgent
    @staticmethod
    def create_symptom_analysis_task(agent: Agent, patient_data: dict, history_analysis: str) -> Task:
        return SymptomTasks.create_symptom_analysis_task(agent, patient_data, history_analysis)

    @staticmethod
    def create_scheduling_task(agent: Agent, patient_data: dict, clinical_assessment: str) -> Task:
        return Task(
            description=f"""
            Coordinate healthcare appointment for: {patient_data['name']}
            
            Clinical Assessment: {clinical_assessment}
            Available Providers: {HEALTHCARE_PROVIDERS}
            
            Determine:
            1. Most appropriate healthcare provider match
            2. Optimal appointment timing based on urgency
            3. Pre-appointment preparation requirements
            4. Follow-up care coordination needs
            
            Provide structured appointment coordination plan.
            """,
            agent=agent,
            expected_output="Detailed appointment coordination with provider matching and timing recommendations"
        )

    @staticmethod
    def create_triage_task(agent: Agent, patient_data: dict, routing_analysis: str) -> Task:
        return Task(
            description=f"""
            Perform comprehensive medical triage for: {patient_data['name']}
            
            Patient Description: {patient_data.get('description', '')}
            Routing Analysis: {routing_analysis}
            
            Conduct triage assessment:
            1. Prioritize patient based on symptom severity and urgency
            2. Determine immediate care needs
            3. Classify urgency level (Emergency/Urgent/Routine)
            4. Recommend appropriate care pathway
            5. Identify any red flag symptoms requiring immediate attention
            
            Provide clear triage decision with priority level and care recommendations.
            """,
            agent=agent,
            expected_output="Comprehensive triage assessment with priority classification and care pathway recommendations"
        )
