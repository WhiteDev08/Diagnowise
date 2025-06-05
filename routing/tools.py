from crewai_tools import tool
import sys
import os
import re
from typing import Dict, Any

# Import the specialized agent systems
try:
    # Import DiagnoWise system
    sys.path.append('./Diagnowise')
    from Diagnowise.crew import diagnowise_crew
    
    # Import EmergencyAgent system  
    sys.path.append('./EmergencyAgent')
    from EmergencyAgent.crew import emergency_crew
    
    # Import HistoryAgent system
    sys.path.append('./HistoryAgent') 
    from HistoryAgent.crew import history_crew
    
    # Import SymptomAgent system
    sys.path.append('./symptom_agent')
    from symptom_agent.crew import symptom_crew
    
except ImportError as e:
    print(f"Warning: Could not import agent systems: {e}")

@tool("route_query_tool")
def route_query_tool(query: str) -> str:
    """
    Main routing tool that analyzes queries and calls the appropriate specialized agent system.
    
    Args:
        query: The user's medical query to be routed
        
    Returns:
        Response from the appropriate specialized agent
    """
    
    # Convert query to lowercase for analysis
    query_lower = query.lower()
    
    # Define routing keywords
    emergency_keywords = [
        'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'difficulty breathing',
        'severe bleeding', 'unconscious', 'emergency', 'urgent', 'critical', 'severe pain',
        'choking', 'overdose', 'poisoning', 'allergic reaction', 'anaphylaxis'
    ]
    
    diagnostic_keywords = [
        'diagnose', 'diagnosis', 'what condition', 'what disease', 'what\'s wrong',
        'medical condition', 'illness', 'disorder', 'treatment options', 'cure'
    ]
    
    history_keywords = [
        'medical history', 'past conditions', 'family history', 'previous illness',
        'medical records', 'health history', 'chronic condition', 'recurring'
    ]
    
    symptom_keywords = [
        'symptoms', 'feeling', 'pain', 'headache', 'fever', 'nausea', 'dizzy',
        'tired', 'fatigue', 'rash', 'cough', 'sore throat', 'stomach ache'
    ]
    
    # Check for emergency situations first (highest priority)
    if any(keyword in query_lower for keyword in emergency_keywords):
        try:
            print("🚨 EMERGENCY DETECTED - Routing to EmergencyAgent...")
            result = emergency_crew.kickoff(inputs={'query': query})
            return f"EMERGENCY RESPONSE:\n{result}"
        except Exception as e:
            return f"Emergency routing failed: {e}. Please seek immediate medical attention by calling emergency services."
    
    # Check for diagnostic queries
    elif any(keyword in query_lower for keyword in diagnostic_keywords):
        try:
            print("🔬 Routing to DiagnoWise for diagnostic analysis...")
            result = diagnowise_crew.kickoff(inputs={'query': query})
            return f"DIAGNOSTIC ANALYSIS:\n{result}"
        except Exception as e:
            return f"Diagnostic routing failed: {e}. Please consult with a healthcare professional."
    
    # Check for medical history queries
    elif any(keyword in query_lower for keyword in history_keywords):
        try:
            print("📋 Routing to HistoryAgent for medical history analysis...")
            result = history_crew.kickoff(inputs={'query': query})
            return f"MEDICAL HISTORY ANALYSIS:\n{result}"
        except Exception as e:
            return f"History analysis routing failed: {e}. Please consult with a healthcare professional."
    
    # Check for symptom-related queries
    elif any(keyword in query_lower for keyword in symptom_keywords):
        try:
            print("🩺 Routing to SymptomAgent for symptom analysis...")
            result = symptom_crew.kickoff(inputs={'query': query})
            return f"SYMPTOM ANALYSIS:\n{result}"
        except Exception as e:
            return f"Symptom analysis routing failed: {e}. Please consult with a healthcare professional."
    
    # Default routing - if no specific keywords detected, route to DiagnoWise as general medical query
    else:
        try:
            print("🏥 General medical query - Routing to DiagnoWise...")
            result = diagnowise_crew.kickoff(inputs={'query': query})
            return f"GENERAL MEDICAL RESPONSE:\n{result}"
        except Exception as e:
            return f"General routing failed: {e}. Please consult with a healthcare professional for your medical query."

def get_routing_decision(query: str) -> Dict[str, Any]:
    """
    Helper function to get routing decision without executing
    
    Args:
        query: The user's medical query
        
    Returns:
        Dictionary with routing decision information
    """
    query_lower = query.lower()
    
    emergency_keywords = ['chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'severe bleeding', 'unconscious', 'emergency']
    diagnostic_keywords = ['diagnose', 'diagnosis', 'what condition', 'what\'s wrong', 'treatment']
    history_keywords = ['medical history', 'past conditions', 'family history', 'medical records']
    symptom_keywords = ['symptoms', 'feeling', 'pain', 'headache', 'fever', 'nausea']
    
    if any(keyword in query_lower for keyword in emergency_keywords):
        return {"agent": "EmergencyAgent", "priority": "HIGH", "reason": "Emergency situation detected"}
    elif any(keyword in query_lower for keyword in diagnostic_keywords):
        return {"agent": "DiagnoWise", "priority": "MEDIUM", "reason": "Diagnostic query detected"}
    elif any(keyword in query_lower for keyword in history_keywords):
        return {"agent": "HistoryAgent", "priority": "LOW", "reason": "Medical history query detected"}
    elif any(keyword in query_lower for keyword in symptom_keywords):
        return {"agent": "SymptomAgent", "priority": "MEDIUM", "reason": "Symptom analysis required"}
    else:
        return {"agent": "DiagnoWise", "priority": "MEDIUM", "reason": "General medical query"}