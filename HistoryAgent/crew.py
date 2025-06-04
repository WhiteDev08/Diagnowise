# crew.py
from crewai import Crew
from agents import medical_history_agent
from task import create_history_analysis_task
from pdf_generator import generate_medical_report_pdf
import json

def main():
    # Get user input
    print("Enter the patient's name:")
    patient_name = input("> ").strip() or "Anonymous Patient"
    
    print("Enter the patient's past medical history (free text):")
    medical_history = input("> ")

    # Create task
    history_task = create_history_analysis_task(medical_history, medical_history_agent)

    # Create Crew and run
    crew = Crew(
        agents=[medical_history_agent], 
        tasks=[history_task],
        verbose=True
    )
    result = crew.kickoff()

    print("\nAnalysis Result:")
    print(result)
    
    # Generate PDF report
    try:
        print(f"\nDEBUG - Raw result type: {type(result)}")
        print(f"DEBUG - Raw result content: {result}")
        
        pdf_filename = generate_medical_report_pdf(result, patient_name)
        print(f"\n✅ PDF Report generated successfully: {pdf_filename}")
    except Exception as e:
        print(f"\n❌ Error generating PDF: {str(e)}")
        print("Please install reportlab: pip install reportlab")

if __name__ == "__main__":
    main()