from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import webbrowser
import tempfile
import threading
import time
import json
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Import the specific agents and tasks from modular structure
try:
    from EmergencyAgent.agents import emergency_agent
    from EmergencyAgent.tasks import create_firstaid_task
    from HistoryAgent.agents import medical_history_agent
    from HistoryAgent.task import create_history_analysis_task
    from SymptomAgent.agents import create_symptom_checker_agent
    from SymptomAgent.task import create_diagnosis_task
    from SymptomAgent.tools import get_diseases_from_neo4j, get_all_symptoms_from_neo4j, close_driver
    from HistoryAgent.pdf_generator import generate_medical_report_pdf
    MODULAR_AGENTS_AVAILABLE = True
    print("✅ Successfully imported modular agents")
except ImportError as e:
    print(f"⚠️ Modular agents not found: {e}")
    print("Using built-in agents as fallback.")
    MODULAR_AGENTS_AVAILABLE = False

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
    print("Please set your OpenAI API key in .env file or environment")
    exit(1)

# Healthcare providers database
HEALTHCARE_PROVIDERS = {
    "cardiology": {
        "name": "Dr. Rajesh Sharma", 
        "email": "rajesh.sharma@cardiaccare.com", 
        "location": "Heart Care Center, Bengaluru", 
        "specializations": ["chest pain", "heart disease", "palpitations", "cardiac"]
    },
    "neurology": {
        "name": "Dr. Priya Nair", 
        "email": "priya.nair@neurocenter.com", 
        "location": "Brain & Spine Clinic, Bengaluru", 
        "specializations": ["headache", "migraine", "dizziness", "neurological"]
    },
    "internal_medicine": {
        "name": "Dr. Amit Singh", 
        "email": "amit.singh@generalhospital.com", 
        "location": "City General Hospital, Bengaluru", 
        "specializations": ["fever", "fatigue", "general health", "internal"]
    },
    "emergency": {
        "name": "Dr. Emergency Team", 
        "email": "emergency@cityhospital.com", 
        "location": "Emergency Department, City Hospital", 
        "specializations": ["emergency", "trauma", "urgent care", "critical"]
    }
}

@tool
def extract_medical_features(medical_history: str) -> dict:
    """Extract medical features from patient history using LLM"""
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.3, max_tokens=400)
        
        prompt = f"""
        Analyze this medical history and extract:
        - Risk factors (diseases, family history, lifestyle)
        - Medication alerts (interactions, allergies)
        - Clinical summary
        
        Medical History: {medical_history}
        
        Return JSON format:
        {{"risk_factors": [...], "medication_alerts": [...], "summary": "..."}}
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Try to extract JSON from response
        response_text = response.content.strip()
        
        # Look for JSON in the response
        if '{' in response_text and '}' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            # Fallback if no JSON found
            result = {
                "risk_factors": ["Analysis could not be parsed"],
                "medication_alerts": ["Please review manually"],
                "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in extract_medical_features: {e}")
        return json.dumps({
            "risk_factors": ["Error processing medical history"],
            "medication_alerts": ["Manual review required"],
            "summary": f"Analysis failed: {str(e)}"
        })

class MedicalReportGenerator:
    @staticmethod
    def generate_pdf_report(patient_data: dict, medical_analysis: str, appointment_details: dict) -> str:
        """Generate comprehensive medical PDF report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_report_{patient_data['name'].replace(' ', '_').replace('/', '_')}_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                       fontSize=18, spaceAfter=30, textColor=colors.darkblue, alignment=1)
            heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                         fontSize=14, spaceAfter=12, textColor=colors.darkred)
            
            # Title and header
            story.append(Paragraph("COMPREHENSIVE MEDICAL REPORT", title_style))
            story.append(Spacer(1, 20))
            
            # Patient information table
            patient_info = [
                ['Patient Name:', patient_data['name']],
                ['Email:', patient_data.get('email', 'Not provided')],
                ['Report Date:', datetime.now().strftime("%B %d, %Y")],
                ['Appointment ID:', appointment_details.get('appointment_id', 'N/A')]
            ]
            
            info_table = Table(patient_info, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT')
            ]))
            story.append(info_table)
            story.append(Spacer(1, 20))
            
            # Current symptoms
            if patient_data.get('symptoms'):
                story.append(Paragraph("PRESENTING SYMPTOMS", heading_style))
                for i, symptom in enumerate(patient_data['symptoms'], 1):
                    story.append(Paragraph(f"{i}. {symptom}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Medical analysis
            try:
                if isinstance(medical_analysis, str) and medical_analysis.strip().startswith('{'):
                    analysis_data = json.loads(medical_analysis)
                else:
                    # If it's not JSON, treat as plain text
                    analysis_data = {"summary": str(medical_analysis)}
                
                # Risk factors
                story.append(Paragraph("IDENTIFIED RISK FACTORS", heading_style))
                if analysis_data.get('risk_factors'):
                    for risk in analysis_data['risk_factors']:
                        story.append(Paragraph(f"• {risk}", styles['Normal']))
                else:
                    story.append(Paragraph("No specific risk factors identified from available information.", styles['Normal']))
                story.append(Spacer(1, 15))
                
                # Medication alerts
                story.append(Paragraph("MEDICATION ALERTS", heading_style))
                if analysis_data.get('medication_alerts'):
                    for alert in analysis_data['medication_alerts']:
                        story.append(Paragraph(f"⚠️ {alert}", styles['Normal']))
                else:
                    story.append(Paragraph("No medication alerts identified.", styles['Normal']))
                story.append(Spacer(1, 15))
                
                # Clinical summary
                story.append(Paragraph("CLINICAL ASSESSMENT", heading_style))
                summary = analysis_data.get('summary', str(medical_analysis)[:500])
                story.append(Paragraph(summary, styles['Normal']))
                
            except Exception as e:
                story.append(Paragraph("MEDICAL ANALYSIS", heading_style))
                story.append(Paragraph(str(medical_analysis), styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Appointment details
            if appointment_details:
                story.append(Paragraph("SCHEDULED APPOINTMENT", heading_style))
                appt_info = [
                    ['Doctor:', appointment_details.get('doctor', 'N/A')],
                    ['Specialty:', appointment_details.get('specialty', 'N/A')],
                    ['Date:', appointment_details.get('date', 'N/A')],
                    ['Time:', appointment_details.get('time', 'N/A')],
                    ['Location:', appointment_details.get('location', 'N/A')]
                ]
                
                appt_table = Table(appt_info, colWidths=[2*inch, 4*inch])
                appt_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10)
                ]))
                story.append(appt_table)
            
            # Footer
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                        textColor=colors.grey, alignment=1)
            story.append(Paragraph("AI-Generated Medical Report - For Healthcare Professional Review", footer_style))
            
            doc.build(story)
            print(f"✅ PDF report generated: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error generating PDF report: {e}")
            # Return a simple text file as fallback
            fallback_filename = f"medical_report_{patient_data['name'].replace(' ', '_')}_{timestamp}.txt"
            with open(fallback_filename, 'w') as f:
                f.write(f"Medical Report for {patient_data['name']}\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Analysis: {medical_analysis}\n")
            return fallback_filename

class HealthcareRoutingSystem:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.openai_api_key, temperature=0.3)
        self.report_generator = MedicalReportGenerator()
    
    def display_main_menu(self):
        """Display the main menu options"""
        print("\n" + "="*60)
        print("           🏥 HEALTHCARE ASSISTANCE SYSTEM")
        print("="*60)
        print("Choose your service:")
        print("1. 🚨 Emergency First Aid")
        print("2. 🔍 Symptom Analysis") 
        print("3. 📋 Medical History Analysis")
        print("4. 📅 Complete Appointment System")  # New option
        print("5. ❌ Exit")
        print("="*60)
    
    def get_user_choice(self):
        """Get and validate user menu choice"""
        while True:
            try:
                choice = input("Enter your choice (1-5): ").strip()
                if choice in ['1', '2', '3', '4', '5']:
                    return int(choice)
                else:
                    print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            except KeyboardInterrupt:
                print("\n\nExiting... Stay safe! 👋")
                return 5
            except Exception:
                print("❌ Invalid input. Please enter a number between 1-5.")
    
    def run_interactive_system(self):
        """Main interactive loop"""
        print("🎯 Welcome to the Healthcare Assistance System!")
        
        while True:
            self.display_main_menu()
            choice = self.get_user_choice()
            
            if choice == 1:
                self._handle_emergency()
            elif choice == 2:
                self._handle_symptom_analysis()
            elif choice == 3:
                self._handle_history_analysis()
            elif choice == 4:
                self._handle_complete_appointment_system()  # New handler
            elif choice == 5:
                print("\n👋 Thank you for using Healthcare Assistance System!")
                print("Stay healthy and safe! 💙")
                break
            
            # Ask if user wants to continue
            if choice != 5:
                print("\n" + "-"*40)
                continue_choice = input("Would you like to use another service? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("\n👋 Thank you for using Healthcare Assistance System!")
                    print("Stay healthy and safe! 💙")
                    break
    
    def _handle_emergency(self):
        """Handle emergency situations"""
        print("\n🚨 EMERGENCY FIRST AID SERVICE")
        print("="*50)
        print("⚠️  DISCLAIMER: This is for informational purposes only.")
        print("⚠️  Call emergency services (911/112) for serious emergencies!")
        print("="*50)
        
        print("\nDescribe your emergency situation in detail:")
        user_input = input("Emergency description: ").strip()
        
        if not user_input:
            print("❌ No emergency description provided. Returning to main menu.")
            return
        
        print("\n🔄 Processing emergency information...")
        
        try:
            # Create the task with user input
            firstaid_task = create_firstaid_task(user_input)
            
            # Create and run the crew
            firstaid_crew = Crew(
                agents=[emergency_agent],
                tasks=[firstaid_task],
                verbose=True
            )
            
            results = firstaid_crew.kickoff()
            print("\n" + "="*50)
            print("🩺 FIRST AID GUIDANCE:")
            print("="*50)
            print(results)
            print("="*50)
            print("⚠️  Remember: Seek professional medical help immediately for serious conditions!")
            
        except Exception as e:
            print(f"❌ Error processing emergency request: {e}")
            print("Please try again or contact emergency services directly.")
    
    def _handle_symptom_analysis(self):
        """Handle symptom analysis"""
        print("\n🔍 SYMPTOM ANALYSIS SERVICE")
        print("="*50)
        print("📝 This service helps analyze your symptoms and suggest possible conditions.")
        print("⚠️  This is not a medical diagnosis - consult a doctor for proper evaluation.")
        print("="*50)
        
        try:
            # Create symptom checker agent
            agent = create_symptom_checker_agent(self.openai_api_key)

            # Get all available symptoms from Neo4j
            print("\n🔄 Loading symptoms from knowledge graph...")
            all_symptoms = get_all_symptoms_from_neo4j()
            
            if not all_symptoms:
                print("❌ Error: Could not retrieve symptoms from database.")
                return
                
            print("✅ Successfully loaded symptom database")
            print("="*50)
            print("📋 Available symptoms (sample):")
            print(", ".join(all_symptoms[:30]))
            if len(all_symptoms) > 30:
                print("... and more")
            print(f"\n📊 Total symptoms in database: {len(all_symptoms)}")
            print("="*50)
            
            print("\n💡 Enter your symptoms separated by commas")
            print("Example: cough, fever, headache, sore throat")
            
            user_symptoms_input = input("Your symptoms: ").strip()
            
            if not user_symptoms_input:
                print("❌ No symptoms entered. Returning to main menu.")
                return
                
            user_symptoms = [s.strip() for s in user_symptoms_input.split(',') if s.strip()]

            # Validate input symptoms against available symptoms
            print(f"\n🔄 Validating {len(user_symptoms)} symptoms...")
            available_symptoms_lower = [s.lower() for s in all_symptoms]
            valid_symptoms = []
            invalid_symptoms = []
            
            for symptom in user_symptoms:
                if symptom.lower() in available_symptoms_lower:
                    valid_symptoms.append(symptom)
                else:
                    invalid_symptoms.append(symptom)
            
            if invalid_symptoms:
                print(f"⚠️  Warning: These symptoms were not found in database: {', '.join(invalid_symptoms)}")
            
            if not valid_symptoms:
                print("❌ No valid symptoms found in database. Please check your input.")
                return
                
            print(f"✅ Processing {len(valid_symptoms)} valid symptoms: {', '.join(valid_symptoms)}")

            # Query Neo4j for top matching diseases
            print("\n🔄 Querying knowledge graph for possible conditions...")
            matched_diseases = get_diseases_from_neo4j(valid_symptoms, top_n=5)

            # Show results to user
            if not matched_diseases:
                print("ℹ️  No conditions matched your symptoms in the knowledge graph.")
            else:
                print("\n📈 Top matches from knowledge graph:")
                for idx, d in enumerate(matched_diseases, 1):
                    print(f"{idx}. {d['disease']} (Matched: {', '.join(d['matched_symptoms'])} | Score: {d['match_count']})")

            # Pass to CrewAI for reasoning and output
            print("\n🤖 Generating AI analysis...")
            diagnosis_task = create_diagnosis_task(agent, valid_symptoms, matched_diseases)
            crew = Crew(
                agents=[agent],
                tasks=[diagnosis_task],
                verbose=True
            )
            
            results = crew.kickoff()
            print("\n" + "="*50)
            print("🩺 SYMPTOM ANALYSIS RESULTS")
            print("="*50)
            print(results)
            print("="*50)
            print("⚠️  Remember: This is not a medical diagnosis. Please consult a healthcare professional.")
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or contact technical support.")
        finally:
            # Clean up database connection
            try:
                close_driver()
            except:
                pass
    
    def _handle_history_analysis(self):
        """Handle medical history analysis"""
        print("\n📋 MEDICAL HISTORY ANALYSIS SERVICE")
        print("="*50)
        print("📊 This service analyzes medical history for patterns and insights.")
        print("🔒 Your information is processed securely and not stored permanently.")
        print("="*50)
        
        try:
            # Get patient information
            print("\nPatient Information:")
            patient_name = input("Enter patient name (or press Enter for 'Anonymous'): ").strip()
            if not patient_name:
                patient_name = "Anonymous Patient"
            
            print(f"\n👤 Processing for: {patient_name}")
            print("\nPlease enter the patient's medical history:")
            print("💡 Include: past conditions, medications, allergies, surgeries, family history, etc.")
            print("📝 Type your history below (press Enter twice when finished):")
            
            medical_history_lines = []
            empty_line_count = 0
            
            while True:
                line = input()
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                else:
                    empty_line_count = 0
                    medical_history_lines.append(line)
            
            medical_history = "\n".join(medical_history_lines).strip()
            
            if not medical_history:
                print("❌ No medical history provided. Returning to main menu.")
                return

            print(f"\n🔄 Analyzing medical history ({len(medical_history)} characters)...")

            # Create task
            history_task = create_history_analysis_task(medical_history, medical_history_agent)

            # Create Crew and run
            crew = Crew(
                agents=[medical_history_agent], 
                tasks=[history_task],
                verbose=True
            )
            result = crew.kickoff()

            print("\n" + "="*50)
            print("📊 MEDICAL HISTORY ANALYSIS RESULTS")
            print("="*50)
            print(result)
            print("="*50)
            
            # Generate PDF report
            print("\n🔄 Generating PDF report...")
            try:
                pdf_filename = generate_medical_report_pdf(result, patient_name)
                print(f"✅ PDF Report generated successfully: {pdf_filename}")
                print("📁 You can find the report in your current directory.")
            except Exception as e:
                print(f"❌ Error generating PDF: {str(e)}")
                print("💡 Note: Please install reportlab if not already installed: pip install reportlab")
                
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or contact technical support.")

    def _handle_complete_appointment_system(self):
        """Handle complete appointment system with comprehensive analysis"""
        print("\n📅 COMPLETE APPOINTMENT SYSTEM")
        print("="*60)
        print("🏥 Comprehensive medical assessment with appointment scheduling")
        print("📋 Includes: History Analysis + Symptom Analysis + Appointment + PDF Report")
        print("="*60)
        
        try:
            # Get patient data
            patient_data = self.get_enhanced_patient_input()
            
            if not patient_data:
                print("❌ Patient data collection failed. Returning to main menu.")
                return
            
            # Process patient with full system
            results = self.process_patient(patient_data)
            
            if results['success']:
                print(f"\n🎉 Complete medical processing finished!")
                print(f"📄 PDF Report: {results['pdf_report_path']}")
                print(f"⚡ Urgency Level: {results['urgency'].upper()}")
                print(f"🏥 Appointment: {results['appointment_details']['date']} at {results['appointment_details']['time']}")
                print(f"👨‍⚕️ Doctor: {results['appointment_details']['doctor']}")
                print(f"📍 Location: {results['appointment_details']['location']}")
            else:
                print(f"❌ Processing failed: {results['error']}")
                
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or contact technical support.")
    
    def get_enhanced_patient_input(self):
        """Enhanced patient input collection with medical history"""
        print("\n👤 PATIENT INFORMATION COLLECTION")
        print("="*50)
        
        name = input("Patient Name: ").strip()
        if not name:
            print("❌ Patient name is required.")
            return None
            
        email = input("📧 Email Address: ").strip()
        if not email:
            print("❌ Email address is required for appointment confirmation.")
            return None
        
        print("\n🩺 Current Symptoms (type 'done' when finished):")
        symptoms = []
        while True:
            symptom = input(f"Symptom {len(symptoms)+1}: ").strip()
            if symptom.lower() == 'done':
                break
            if symptom:
                symptoms.append(symptom)
        
        if not symptoms:
            print("❌ At least one symptom is required.")
            return None
        
        print("\n📋 Medical History (optional - press Enter to skip):")
        medical_history = input("Previous conditions, medications, family history: ").strip()
        if not medical_history:
            medical_history = "No significant medical history reported."
        
        return {
            'name': name, 
            'email': email, 
            'symptoms': symptoms,
            'medical_history': medical_history
        }
    
    def create_fallback_medical_history_agent(self) -> Agent:
        """Fallback medical history agent when modular agents are not available"""
        return Agent(
            role="Medical History Analyzer",
            goal="Analyze patient medical history and extract comprehensive medical insights",
            backstory="""You are an expert medical AI that analyzes patient histories, identifies risk factors, 
                       medication interactions, and provides clinical summaries for healthcare providers. 
                       You have extensive knowledge of medical conditions, drug interactions, and clinical patterns.""",
            tools=[extract_medical_features],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_fallback_symptom_analyzer(self) -> Agent:
        """Fallback symptom analyzer when modular agents are not available"""
        return Agent(
            role="Clinical Symptom Analyzer",
            goal="Analyze current symptoms and provide medical assessment with urgency classification",
            backstory="""You are an advanced diagnostic AI that evaluates presenting symptoms, determines urgency levels, 
                       and recommends appropriate specialist referrals based on clinical presentation. You can assess 
                       symptom combinations and provide differential diagnoses.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_appointment_scheduler(self) -> Agent:
        return Agent(
            role="Healthcare Appointment Coordinator",
            goal="Match patients with appropriate specialists and optimize appointment scheduling",
            backstory="""You are an intelligent scheduling system that considers symptom urgency, specialist availability, 
                       and patient needs to coordinate optimal healthcare appointments. You understand medical specialties 
                       and can match symptoms to appropriate healthcare providers.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_medical_history_task_enhanced(self, agent: Agent, patient_data: dict) -> Task:
        """Enhanced medical history task that works with both modular and fallback agents"""
        medical_history = patient_data.get('medical_history', 'No previous medical history provided.')
        
        if MODULAR_AGENTS_AVAILABLE:
            try:
                # Use modular task creation
                return create_history_analysis_task(medical_history, agent)
            except Exception as e:
                print(f"⚠️ Error using modular history task, falling back: {e}")
        
        # Use fallback task creation
        return Task(
            description=f"""
            Analyze comprehensive medical profile for: {patient_data['name']}
            
            Current Symptoms: {', '.join(patient_data.get('symptoms', []))}
            Medical History: {medical_history}
            
            Please provide detailed analysis including:
            1. Risk factor identification and assessment
            2. Medication alerts and contraindications  
            3. Clinical correlation between history and current symptoms
            4. Comprehensive medical summary with recommendations
            
            Focus on clinically relevant insights that would help healthcare providers.
            """,
            agent=agent,
            expected_output="Comprehensive medical analysis with risk factors, alerts, and clinical summary"
        )
    
    def create_symptom_analysis_task_enhanced(self, agent: Agent, patient_data: dict, history_analysis: str = "") -> Task:
        """Enhanced symptom analysis task"""
        symptoms = patient_data.get('symptoms', [])
        
        if MODULAR_AGENTS_AVAILABLE:
            try:
                # Get diseases from Neo4j if available
                matched_diseases = get_diseases_from_neo4j(symptoms, top_n=5)
                return create_diagnosis_task(agent, symptoms, matched_diseases)
            except Exception as e:
                print(f"⚠️ Error using modular symptom analysis, falling back: {e}")
        
        # Fallback task creation
        return Task(
            description=f"""
            Clinical symptom assessment for: {patient_data['name']}
            
            Presenting Symptoms: {', '.join(symptoms)}
            Medical History Context: {history_analysis[:500] if history_analysis else 'Not available'}
            
            Please provide:
            1. Differential diagnosis considerations
            2. Urgency classification (EMERGENCY/URGENT/ROUTINE)
            3. Recommended specialist type and rationale
            4. Clinical correlation with medical history (if available)
            5. Immediate care recommendations
            
            Consider symptom combinations and potential underlying conditions.
            Always err on the side of caution for urgency assessment.
            """,
            agent=agent,
            expected_output="Clinical assessment with urgency level and specialist recommendation"
        )
    
    def create_scheduling_task(self, agent: Agent, patient_data: dict, clinical_assessment: str) -> Task:
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
    
    def determine_specialty(self, symptoms: list) -> str:
        symptom_text = " ".join(symptoms).lower()
        for specialty, provider in HEALTHCARE_PROVIDERS.items():
            if any(spec in symptom_text for spec in provider['specializations']):
                return specialty
        return "internal_medicine"
    
    def generate_appointment_details(self, specialty: str, patient_name: str, urgency: str = "routine") -> dict:
        provider = HEALTHCARE_PROVIDERS.get(specialty, HEALTHCARE_PROVIDERS['internal_medicine'])
        
        if urgency.lower() == "emergency":
            date = datetime.now().strftime('%Y-%m-%d')
            time = "IMMEDIATE - Emergency Department"
        elif urgency.lower() == "urgent":
            date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            time = "09:00 AM"
        else:
            date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
            time = "10:00 AM"
        
        return {
            'patient': patient_name,
            'doctor': provider['name'],
            'specialty': specialty.replace('_', ' ').title(),
            'date': date,
            'time': time,
            'location': provider['location'],
            'doctor_email': provider['email'],
            'appointment_id': f"APPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def create_web_email_interface(self, patient_email: str, subject: str, body: str, pdf_path: str = None) -> str:
        """Enhanced email interface with PDF attachment support"""
        
        pdf_attachment_html = ""
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_base64 = base64.b64encode(f.read()).decode()
            pdf_attachment_html = f'''
            <div class="attachment-section">
                <h3>📎 Medical Report Attachment</h3>
                <a href="data:application/pdf;base64,{pdf_base64}" download="medical_report.pdf" 
                   class="btn btn-success">📄 Download Medical Report</a>
            </div>'''
        
        body_html = body.replace('\n', '<br>').replace('"', '&quot;')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Healthcare Email System</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); 
               min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 15px; 
                     box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(45deg, #2196F3, #21CBF3); color: white; padding: 30px; text-align: center; }}
        .content {{ padding: 30px; }}
        .email-preview {{ background: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; 
                         padding: 20px; margin: 20px 0; max-height: 400px; overflow-y: auto; }}
        .attachment-section {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0; 
                              border: 2px solid #4CAF50; }}
        .form-group {{ margin: 15px 0; }}
        label {{ display: block; margin-bottom: 5px; font-weight: 600; color: #333; }}
        input {{ width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; }}
        .btn {{ padding: 12px 20px; border: none; border-radius: 8px; font-weight: 600; 
               cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; }}
        .btn-primary {{ background: #2196F3; color: white; }}
        .btn-success {{ background: #4CAF50; color: white; }}
        .btn-warning {{ background: #FF9800; color: white; }}
        .btn-group {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Advanced Healthcare Email System</h1>
            <p>Comprehensive Medical Report & Appointment Management</p>
        </div>
        
        <div class="content">
            <div class="form-group">
                <label>📧 Patient Email:</label>
                <input type="email" value="{patient_email}" readonly>
            </div>
            
            <div class="form-group">
                <label>📋 Subject:</label>
                <input type="text" value="{subject}" readonly>
            </div>
            
            {pdf_attachment_html}
            
            <div class="form-group">
                <label>💌 Email Content:</label>
                <div class="email-preview">{body_html}</div>
            </div>
            
            <div class="btn-group">
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to={patient_email}&su={subject}&body={body.replace(' ', '%20').replace('\n', '%0A')}" 
                   target="_blank" class="btn btn-primary">📧 Send via Gmail</a>
                <a href="mailto:{patient_email}?subject={subject}&body={body}" class="btn btn-warning">📧 Default Email</a>
                <button onclick="copyContent()" class="btn btn-success">📋 Copy All Content</button>
            </div>
        </div>
    </div>
    
    <script>
        function copyContent() {{
            const content = `To: {patient_email}\\nSubject: {subject}\\n\\n{body}`;
            navigator.clipboard.writeText(content).then(() => {{
                alert('📋 Email content copied to clipboard!');
            }});
        }}
        setTimeout(() => {{
            document.querySelector('a[href*="gmail"]').click();
        }}, 2000);
    </script>
</body>
</html>"""
        return html_content
    
    def process_patient(self, patient_data: dict) -> dict:
        """Enhanced patient processing with comprehensive medical analysis"""
        print(f"\n🚀 Processing comprehensive medical case for: {patient_data['name']}")
        print("=" * 60)
        
        try:
            # Create specialized agents
            print("🤖 Initializing medical AI agents...")
            history_agent = self.create_medical_history_agent()
            symptom_agent = self.create_symptom_analyzer()
            scheduler_agent = self.create_appointment_scheduler()c
            
            # Step 1: Medical History Analysis
            print("\n📋 STEP 1: Comprehensive Medical History Analysis")
            history_task = self.create_medical_history_task(history_agent, patient_data)
            history_crew = Crew(agents=[history_agent], tasks=[history_task], process=Process.sequential)
            history_analysis = history_crew.kickoff()
            
            # Step 2: Current Symptom Analysis
            print("\n🩺 STEP 2: Clinical Symptom Assessment")
            symptom_task = self.create_symptom_analysis_task(symptom_agent, patient_data, str(history_analysis))
            symptom_crew = Crew(agents=[symptom_agent], tasks=[symptom_task], process=Process.sequential)
            clinical_assessment = symptom_crew.kickoff()
            
            # Step 3: Appointment Coordination
            print("\n📅 STEP 3: Healthcare Appointment Coordination")
            scheduling_task = self.create_scheduling_task(scheduler_agent, patient_data, str(clinical_assessment))
            scheduling_crew = Crew(agents=[scheduler_agent], tasks=[scheduling_task], process=Process.sequential)
            appointment_coordination = scheduling_crew.kickoff()
            
            # Generate appointment details
            specialty = self.determine_specialty(patient_data['symptoms'])
            urgency = "routine"
            
            # Extract urgency from clinical assessment
            assessment_text = str(clinical_assessment).upper()
            if "EMERGENCY" in assessment_text:
                urgency = "emergency"
            elif "URGENT" in assessment_text:
                urgency = "urgent"
            
            appointment_details = self.generate_appointment_details(specialty, patient_data['name'], urgency)
            
            # Generate comprehensive PDF report
            print("\n📄 STEP 4: Generating Comprehensive Medical Report")
            pdf_path = self.report_generator.generate_pdf_report(
                patient_data, str(history_analysis), appointment_details
            )
            
            # Create email content
            email_subject = f"Medical Report & Appointment - {patient_data['name']} - {appointment_details['appointment_id']}"
            email_body = f"""Dear {patient_data['name']},

Your comprehensive medical assessment has been completed. Please find attached your detailed medical report.

APPOINTMENT CONFIRMED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👨‍⚕️ Doctor: {appointment_details['doctor']}
🏥 Specialty: {appointment_details['specialty']}
📅 Date: {appointment_details['date']}
⏰ Time: {appointment_details['time']}
📍 Location: {appointment_details['location']}
🆔 Appointment ID: {appointment_details['appointment_id']}

IMPORTANT INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Please review the attached medical report before your appointment
• Arrive 15 minutes early with valid ID and insurance
• Bring all current medications and previous medical records
• Prepare questions based on the medical analysis provided

MEDICAL SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Symptoms: {', '.join(patient_data['symptoms'])}
Assessment: {str(clinical_assessment)[:200]}...

For questions: {appointment_details['doctor_email']}
Emergency: Call 108

Best regards,
AI Healthcare System
Bengaluru Medical Network"""
            
            # Launch enhanced web email interface
            print("\n🌐 STEP 5: Launching Enhanced Email Interface")
            html_content = self.create_web_email_interface(patient_data['email'], email_subject, email_body, pdf_path)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                webbrowser.open(f'file://{f.name}')
            
            print("✅ COMPREHENSIVE MEDICAL PROCESSING COMPLETED!")
            print(f"📄 Medical report generated: {pdf_path}")
            print("🌐 Enhanced email interface launched with PDF attachment support")
            
            return {
                'success': True,
                'patient_info': patient_data,
                'medical_history_analysis': str(history_analysis),
                'clinical_assessment': str(clinical_assessment),
                'appointment_details': appointment_details,
                'pdf_report_path': pdf_path,
                'urgency': urgency
            }
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}

def get_enhanced_patient_input():
    """Enhanced patient input collection with medical history"""
    print("\n🏥 ADVANCED HEALTHCARE AI SYSTEM")
    print("=" * 50)
    
    name = input("👤 Patient Name: ").strip()
    email = input("📧 Email Address: ").strip()
    
    print("\n🩺 Current Symptoms (type 'done' when finished):")
    symptoms = []
    while True:
        symptom = input(f"Symptom {len(symptoms)+1}: ").strip()
        if symptom.lower() == 'done':
            break
        if symptom:
            symptoms.append(symptom)
    
    print("\n📋 Medical History (optional - press Enter to skip):")
    medical_history = input("Previous conditions, medications, family history: ").strip()
    if not medical_history:
        medical_history = "No significant medical history reported."
    
    return {
        'name': name, 
        'email': email, 
        'symptoms': symptoms,
        'medical_history': medical_history
    }
def main():
    """Main function to run the healthcare system"""
    try:
        # Initialize the healthcare routing system
        healthcare_system = HealthcareRoutingSystem()
        
        # Run the interactive system
        healthcare_system.run_interactive_system()
        
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ System error: {e}")
        print("Please contact technical support.")


# Run the system
if __name__ == "__main__":
    main()

    
