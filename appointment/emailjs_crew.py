from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Enhanced Healthcare providers database
HEALTHCARE_PROVIDERS = {
    "cardiology": {
        "name": "Dr. Rajesh Sharma", 
        "email": "rajesh.sharma@cardiaccare.com", 
        "location": "Heart Care Center, Bengaluru", 
        "specializations": ["chest pain", "heart disease", "palpitations", "cardiac", "hypertension"],
        "available_slots": ["9:00 AM", "10:00 AM", "2:00 PM", "3:00 PM"]
    },
    "neurology": {
        "name": "Dr. Priya Nair", 
        "email": "priya.nair@neurocenter.com", 
        "location": "Brain & Spine Clinic, Bengaluru", 
        "specializations": ["headache", "migraine", "dizziness", "neurological", "seizure"],
        "available_slots": ["9:30 AM", "11:00 AM", "2:30 PM", "4:00 PM"]
    },
    "internal_medicine": {
        "name": "Dr. Amit Singh", 
        "email": "amit.singh@generalhospital.com", 
        "location": "City General Hospital, Bengaluru", 
        "specializations": ["fever", "fatigue", "general health", "internal", "diabetes"],
        "available_slots": ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM"]
    },
    "orthopedics": {
        "name": "Dr. Kavya Reddy",
        "email": "kavya.reddy@boneclinic.com",
        "location": "Bone & Joint Clinic, Bengaluru",
        "specializations": ["joint pain", "back pain", "fracture", "arthritis", "sports injury"],
        "available_slots": ["10:00 AM", "11:30 AM", "3:00 PM", "4:30 PM"]
    }
}

@tool
def extract_comprehensive_medical_features(medical_history: str, symptoms: list, urgency: str) -> dict:
    """Enhanced medical feature extraction with urgency assessment"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.2, max_tokens=600)
    
    prompt = f"""
    Perform comprehensive medical analysis:
    
    Current Symptoms: {', '.join(symptoms)}
    Medical History: {medical_history}
    Urgency Level: {urgency}
    
    Analyze and extract:
    1. Risk factors (diseases, family history, lifestyle, age-related)
    2. Medication alerts (interactions, allergies, contraindications)
    3. Differential diagnosis possibilities
    4. Urgency assessment and triage level
    5. Recommended specialist type
    6. Clinical correlation between symptoms and history
    7. Immediate care recommendations
    
    Return comprehensive JSON format:
    {{
        "risk_factors": [...],
        "medication_alerts": [...],
        "differential_diagnosis": [...],
        "urgency_assessment": "...",
        "recommended_specialist": "...",
        "clinical_correlation": "...",
        "immediate_care": [...],
        "summary": "...",
        "emailjs_summary": "Brief summary for email delivery"
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "risk_factors": ["Analysis unavailable"],
            "medication_alerts": ["Please consult healthcare provider"],
            "differential_diagnosis": ["Requires professional evaluation"],
            "urgency_assessment": urgency,
            "recommended_specialist": "General Medicine",
            "clinical_correlation": "Unable to analyze",
            "immediate_care": ["Seek medical attention"],
            "summary": f"Analysis error: {str(e)}",
            "emailjs_summary": "Medical analysis completed. Please consult with healthcare provider."
        })

@tool
def schedule_optimal_appointment(symptoms: list, urgency: str, preferred_date: str, preferred_time: str) -> dict:
    """Enhanced appointment scheduling with preference consideration for EmailJS"""
    
    # Determine best specialty based on symptoms
    symptom_text = " ".join(symptoms).lower()
    best_specialty = "internal_medicine"  # default
    
    for specialty, provider in HEALTHCARE_PROVIDERS.items():
        if any(spec in symptom_text for spec in provider['specializations']):
            best_specialty = specialty
            break
    
    provider = HEALTHCARE_PROVIDERS[best_specialty]
    
    # Calculate appointment timing based on urgency and preferences
    if urgency == "emergency":
        date = datetime.now().strftime('%Y-%m-%d')
        time = "IMMEDIATE - Emergency Department"
        location = "Emergency Department"
    elif urgency == "urgent":
        # Try to accommodate within 1-2 days
        target_date = datetime.now() + timedelta(days=1)
        if preferred_date:
            pref_date = datetime.strptime(preferred_date, '%Y-%m-%d')
            if pref_date <= datetime.now() + timedelta(days=2):
                target_date = pref_date
        
        date = target_date.strftime('%Y-%m-%d')
        time = preferred_time if preferred_time in provider['available_slots'] else provider['available_slots'][0]
        location = provider['location']
    else:  # routine
        # Try to accommodate preferred date/time
        if preferred_date:
            target_date = datetime.strptime(preferred_date, '%Y-%m-%d')
        else:
            target_date = datetime.now() + timedelta(days=3)
        
        date = target_date.strftime('%Y-%m-%d')
        time = preferred_time if preferred_time in provider['available_slots'] else provider['available_slots'][0]
        location = provider['location']
    
    return {
        'specialty': best_specialty,
        'doctor': provider['name'],
        'doctor_email': provider['email'],
        'date': date,
        'time': time,
        'location': location,
        'appointment_id': f"APPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'scheduling_rationale': f"Selected {best_specialty} based on symptoms. Urgency: {urgency}.",
        'emailjs_ready': True
    }

class EmailJSMedicalReportGenerator:
    @staticmethod
    def generate_emailjs_pdf_report(patient_data: dict, medical_analysis: str, appointment_details: dict) -> str:
        """Generate PDF report optimized for EmailJS delivery"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emailjs_medical_report_{patient_data['name'].replace(' ', '_')}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # EmailJS optimized styles
        title_style = ParagraphStyle('EmailJSTitle', parent=styles['Heading1'], 
                                   fontSize=20, spaceAfter=30, textColor=colors.darkblue, 
                                   alignment=1, fontName='Helvetica-Bold')
        heading_style = ParagraphStyle('EmailJSHeading', parent=styles['Heading2'], 
                                     fontSize=16, spaceAfter=15, textColor=colors.darkred,
                                     fontName='Helvetica-Bold')
        
        # Title and header
        story.append(Paragraph("AI MEDICAL ANALYSIS REPORT", title_style))
        story.append(Paragraph("Delivered via EmailJS Healthcare System", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Patient information table
        patient_info = [
            ['Patient Name:', patient_data['name']],
            ['Email:', patient_data['email']],
            ['Phone:', patient_data.get('phone', 'Not provided')],
            ['Report Date:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ['Appointment ID:', appointment_details['appointment_id']],
            ['Delivery Method:', 'EmailJS Automatic Delivery']
        ]
        
        info_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(info_table)
        story.append(Spacer(1, 25))
        
        # Current symptoms section
        story.append(Paragraph("PRESENTING SYMPTOMS", heading_style))
        for i, symptom in enumerate(patient_data['symptoms'], 1):
            story.append(Paragraph(f"• {symptom}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Medical analysis optimized for EmailJS
        try:
            analysis_data = json.loads(medical_analysis) if isinstance(medical_analysis, str) else medical_analysis
            
            # EmailJS summary
            if analysis_data.get('emailjs_summary'):
                story.append(Paragraph("EMAILJS DELIVERY SUMMARY", heading_style))
                story.append(Paragraph(analysis_data['emailjs_summary'], styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Risk factors
            story.append(Paragraph("IDENTIFIED RISK FACTORS", heading_style))
            if analysis_data.get('risk_factors'):
                for risk in analysis_data['risk_factors']:
                    story.append(Paragraph(f"⚠️ {risk}", styles['Normal']))
            else:
                story.append(Paragraph("No specific risk factors identified.", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Clinical summary
            story.append(Paragraph("CLINICAL ASSESSMENT", heading_style))
            summary = analysis_data.get('summary', 'No summary available')
            story.append(Paragraph(summary, styles['Normal']))
            
        except Exception as e:
            story.append(Paragraph("MEDICAL ANALYSIS", heading_style))
            story.append(Paragraph(str(medical_analysis), styles['Normal']))
        
        story.append(Spacer(1, 25))
        
        # Appointment details
        story.append(Paragraph("SCHEDULED APPOINTMENT", heading_style))
        appt_info = [
            ['Doctor:', appointment_details['doctor']],
            ['Specialty:', appointment_details.get('specialty', 'General Medicine')],
            ['Date:', appointment_details['date']],
            ['Time:', appointment_details['time']],
            ['Location:', appointment_details['location']],
            ['EmailJS Delivery:', 'Automatic Email Sent']
        ]
        
        appt_table = Table(appt_info, colWidths=[2*inch, 4*inch])
        appt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(appt_table)
        
        # EmailJS footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle('EmailJSFooter', parent=styles['Normal'], fontSize=9, 
                                    textColor=colors.grey, alignment=1)
        story.append(Paragraph("AI-Generated Medical Report - Delivered via EmailJS", footer_style))
        story.append(Paragraph("This report was automatically sent to your email using EmailJS technology", footer_style))
        
        doc.build(story)
        return filename

class EmailJSHealthcareCrewAI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, temperature=0.2)
        self.report_generator = EmailJSMedicalReportGenerator()
    
    def create_enhanced_medical_history_agent(self) -> Agent:
        return Agent(
            role="Senior Medical History Analyst (EmailJS Ready)",
            goal="Perform comprehensive analysis of patient medical history optimized for EmailJS delivery",
            backstory="Expert medical AI specialized in analyzing patient histories and preparing "
                     "comprehensive medical insights for EmailJS email delivery. Focuses on creating "
                     "clear, actionable medical summaries suitable for email communication.",
            tools=[extract_comprehensive_medical_features],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_advanced_symptom_analyzer(self) -> Agent:
        return Agent(
            role="Advanced Clinical Symptom Diagnostician (EmailJS Compatible)",
            goal="Analyze symptoms with differential diagnosis optimized for EmailJS delivery",
            backstory="Highly advanced diagnostic AI that evaluates presenting symptoms and creates "
                     "comprehensive assessments suitable for EmailJS email delivery. Specializes in "
                     "clear, patient-friendly medical communication.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_intelligent_appointment_scheduler(self) -> Agent:
        return Agent(
            role="Intelligent Healthcare Appointment Coordinator (EmailJS Enabled)",
            goal="Optimize appointment scheduling for EmailJS notification delivery",
            backstory="Advanced scheduling AI that coordinates appointments and prepares detailed "
                     "scheduling information for EmailJS email delivery. Ensures all appointment "
                     "details are properly formatted for email communication.",
            tools=[schedule_optimal_appointment],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    def create_comprehensive_medical_analysis_task(self, agent: Agent, patient_data: dict) -> Task:
        medical_history = patient_data.get('medical_history', 'No previous medical history provided.')
        urgency = patient_data.get('urgency_level', 'routine')
        
        return Task(
            description=f"""
            Perform comprehensive medical analysis optimized for EmailJS delivery for: {patient_data['name']}
            
            PATIENT PROFILE:
            - Current Symptoms: {', '.join(patient_data['symptoms'])}
            - Medical History: {medical_history}
            - Urgency Level: {urgency}
            - Email: {patient_data['email']} (for EmailJS delivery)
            
            ANALYSIS FOR EMAILJS DELIVERY:
            1. Comprehensive risk factor identification
            2. Medication alerts and contraindications
            3. Differential diagnosis with clear explanations
            4. Clinical correlation suitable for email communication
            5. Urgency assessment with clear rationale
            6. Specialist recommendation with justification
            7. Immediate care recommendations
            8. EmailJS-optimized summary for email delivery
            
            Use extract_comprehensive_medical_features tool and ensure output is EmailJS-ready.
            """,
            agent=agent,
            expected_output="Comprehensive medical analysis in JSON format optimized for EmailJS email delivery"
        )
    
    def create_advanced_symptom_assessment_task(self, agent: Agent, patient_data: dict, history_analysis: str) -> Task:
        return Task(
            description=f"""
            Advanced clinical symptom assessment for EmailJS delivery: {patient_data['name']}
            
            CLINICAL PRESENTATION:
            - Presenting Symptoms: {', '.join(patient_data['symptoms'])}
            - Urgency Level: {patient_data.get('urgency_level', 'routine')}
            - Medical History Analysis: {history_analysis}
            - EmailJS Target: {patient_data['email']}
            
            ASSESSMENT FOR EMAILJS:
            1. Clear symptom analysis suitable for email communication
            2. Differential diagnosis with patient-friendly explanations
            3. Urgency classification with clear rationale
            4. Specialist recommendations with justification
            5. Clinical correlation in accessible language
            6. Care recommendations suitable for email delivery
            
            Ensure all output is clear and suitable for EmailJS email delivery.
            """,
            agent=agent,
            expected_output="Clinical assessment optimized for EmailJS email communication"
        )
    
    def create_intelligent_scheduling_task(self, agent: Agent, patient_data: dict, clinical_assessment: str) -> Task:
        preferred_date = patient_data.get('preferred_date', '')
        preferred_time = patient_data.get('preferred_time', '')
        urgency = patient_data.get('urgency_level', 'routine')
        
        return Task(
            description=f"""
            Intelligent appointment coordination for EmailJS delivery: {patient_data['name']}
            
            SCHEDULING FOR EMAILJS:
            - Clinical Assessment: {clinical_assessment}
            - Patient Preferences: Date: {preferred_date}, Time: {preferred_time}
            - Urgency Level: {urgency}
            - EmailJS Target: {patient_data['email']}
            - Available Providers: {HEALTHCARE_PROVIDERS}
            
            EMAILJS SCHEDULING OPTIMIZATION:
            1. Match with most appropriate healthcare provider
            2. Consider urgency and patient preferences
            3. Prepare appointment details for EmailJS delivery
            4. Create clear scheduling rationale for email
            5. Ensure all details are EmailJS-compatible
            
            Use schedule_optimal_appointment tool for EmailJS-ready scheduling.
            """,
            agent=agent,
            expected_output="Appointment coordination details optimized for EmailJS delivery"
        )
    
    def process_patient_for_emailjs(self, patient_data: dict) -> dict:
        """Process patient data and prepare for EmailJS delivery"""
        print(f"\n🚀 Starting EmailJS-optimized medical processing for: {patient_data['name']}")
        print("=" * 70)
        
        try:
            # Create EmailJS-optimized agents
            print("🤖 Initializing EmailJS-ready medical AI agents...")
            history_agent = self.create_enhanced_medical_history_agent()
            symptom_agent = self.create_advanced_symptom_analyzer()
            scheduler_agent = self.create_intelligent_appointment_scheduler()
            
            # Step 1: Medical History Analysis for EmailJS
            print("\n📋 STEP 1: Medical History Analysis (EmailJS Ready)")
            history_task = self.create_comprehensive_medical_analysis_task(history_agent, patient_data)
            history_crew = Crew(agents=[history_agent], tasks=[history_task], process=Process.sequential)
            history_analysis = history_crew.kickoff()
            print("✅ EmailJS-ready medical history analysis completed")
            
            # Step 2: Symptom Assessment for EmailJS
            print("\n🩺 STEP 2: Symptom Assessment (EmailJS Compatible)")
            symptom_task = self.create_advanced_symptom_assessment_task(symptom_agent, patient_data, str(history_analysis))
            symptom_crew = Crew(agents=[symptom_agent], tasks=[symptom_task], process=Process.sequential)
            clinical_assessment = symptom_crew.kickoff()
            print("✅ EmailJS-compatible clinical assessment completed")
            
            # Step 3: Appointment Coordination for EmailJS
            print("\n📅 STEP 3: Appointment Coordination (EmailJS Enabled)")
            scheduling_task = self.create_intelligent_scheduling_task(scheduler_agent, patient_data, str(clinical_assessment))
            scheduling_crew = Crew(agents=[scheduler_agent], tasks=[scheduling_task], process=Process.sequential)
            appointment_coordination = scheduling_crew.kickoff()
            print("✅ EmailJS-enabled appointment coordination completed")
            
            # Step 4: Generate EmailJS-Optimized PDF Report
            print("\n📄 STEP 4: Generating EmailJS-Optimized PDF Report")
            
            # Get structured appointment details
            appointment_details = schedule_optimal_appointment(
                patient_data['symptoms'], 
                patient_data.get('urgency_level', 'routine'), 
                patient_data.get('preferred_date', ''),
                patient_data.get('preferred_time', '')
            )
            
            if isinstance(appointment_details, str):
                try:
                    appointment_details = json.loads(appointment_details)
                except:
                    appointment_details = {
                        'doctor': 'Dr. Amit Singh',
                        'specialty': 'Internal Medicine',
                        'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                        'time': '10:00 AM',
                        'location': 'City General Hospital, Bengaluru',
                        'doctor_email': 'amit.singh@generalhospital.com',
                        'appointment_id': f"EMAILJS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'emailjs_ready': True
                    }
            
            pdf_path = self.report_generator.generate_emailjs_pdf_report(
                patient_data, str(history_analysis), appointment_details
            )
            print(f"✅ EmailJS-optimized PDF report generated: {pdf_path}")
            
            # Prepare EmailJS data
            emailjs_data = {
                'patient_email': patient_data['email'],
                'patient_name': patient_data['name'],
                'doctor_name': appointment_details['doctor'],
                'appointment_date': appointment_details['date'],
                'appointment_time': appointment_details['time'],
                'appointment_location': appointment_details['location'],
                'appointment_id': appointment_details['appointment_id'],
                'medical_summary': f"Symptoms: {', '.join(patient_data['symptoms'])}",
                'report_path': pdf_path,
                'emailjs_ready': True
            }
            
            print("\n🎉 EMAILJS PROCESSING COMPLETED!")
            print(f"📧 EmailJS data prepared for: {patient_data['email']}")
            print(f"📄 Report ready: {pdf_path}")
            
            return {
                'success': True,
                'patient_info': patient_data,
                'medical_history_analysis': str(history_analysis),
                'clinical_assessment': str(clinical_assessment),
                'appointment_coordination': str(appointment_coordination),
                'appointment_details': appointment_details,
                'pdf_report_path': pdf_path,
                'emailjs_data': emailjs_data,
                'urgency': patient_data.get('urgency_level', 'routine'),
                'processing_summary': 'All CrewAI agents utilized for EmailJS delivery'
            }
            
        except Exception as e:
            print(f"❌ EmailJS processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}

# Test the EmailJS system
if __name__ == "__main__":
    print("🚀 Testing EmailJS Healthcare CrewAI System")
    
    test_patient = {
        'name': 'Jane Smith',
        'email': 'jane.smith@example.com',
        'phone': '+91-9876543210',
        'symptoms': ['headache', 'dizziness', 'nausea'],
        'medical_history': 'No significant medical history',
        'urgency_level': 'routine',
        'preferred_date': '2024-01-20',
        'preferred_time': '10:00 AM'
    }
    
    healthcare_system = EmailJSHealthcareCrewAI()
    results = healthcare_system.process_patient_for_emailjs(test_patient)
    
    if results['success']:
        print("\n✅ EmailJS test completed successfully!")
        print(f"📧 EmailJS data ready: {results['emailjs_data']['emailjs_ready']}")
    else:
        print(f"\n❌ EmailJS test failed: {results['error']}")
