from crewai import Task

def create_routing_task(user_query: str):
    """Create a routing task that analyzes and routes user queries to appropriate agents"""
    return Task(
        description=f"""
        Analyze the following user query and route it to the most appropriate specialized agent:
        
        USER QUERY: "{user_query}"
        
        Your task is to:
        1. Analyze the nature and urgency of the query
        2. Determine which specialized agent should handle this query:
           
           Route to EMERGENCY AGENT if query contains:
           - Emergency keywords: chest pain, can't breathe, severe bleeding, unconscious, heart attack, stroke
           - Life-threatening situations or urgent medical crises
           - Severe pain or distress indicators
           
           Route to DIAGNOWISE if query contains:
           - Diagnostic requests: "what condition do I have", "diagnose", "what's wrong with me"
           - Medical condition analysis and disease information
           - Treatment options and medical advice requests
           
           Route to HISTORY AGENT if query contains:
           - Medical history analysis: "medical records", "past conditions", "family history"
           - Pattern recognition in health data
           - Historical health information requests
           
           Route to SYMPTOM AGENT if query contains:
           - Symptom analysis: "I have symptoms", "feeling", specific symptoms described
           - Symptom interpretation and correlation
           - General symptom-related questions
        
        3. Execute the routing by calling the appropriate agent's system
        4. Return the response from the selected agent
        
        IMPORTANT: Always prioritize emergency situations - if there's any indication of emergency, route to EmergencyAgent immediately.
        """,
        expected_output="The response from the appropriate specialized agent after successful routing and query processing."
    )