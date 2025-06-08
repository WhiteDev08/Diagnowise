from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
import json
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Email configuration - Add these to your .env file
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # Your email
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # App password for Gmail

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
    """Enhanced medical feature extraction with urgency assessment using Groq"""
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile", temperature=0.2, max_tokens=600)
    
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
        "summary": "..."
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
            "summary": f"Analysis error: {str(e)}"
        })

@tool
def schedule_optimal_appointment(symptoms: list, urgency: str, preferred_date: str, preferred_time: str) -> dict:
    """Enhanced appointment scheduling with preference consideration"""
    
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
        'scheduling_rationale': f"Selected {best_specialty} based on symptoms. Urgency: {urgency}."
    }

class EnhancedMedicalReportGenerator:
    @staticmethod
    def generate_comprehensive_pdf_report(patient_data: dict, medical_analysis: str, appointment_details: dict) -> str:
        """Generate enhanced comprehensive medical PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_medical_report_{patient_data['name'].replace(' ', '_')}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Enhanced custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                   fontSize=20, spaceAfter=30, textColor=colors.darkblue, 
                                   alignment=1, fontName='Helvetica-Bold')
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                     fontSize=16, spaceAfter=15, textColor=colors.darkred,
                                     fontName='Helvetica-Bold')
        subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'],
                                        fontSize=12, spaceAfter=10, textColor=colors.darkgreen,
                                        fontName='Helvetica-Bold')
        
        # Title and header
        story.append(Paragraph("COMPREHENSIVE AI MEDICAL ANALYSIS REPORT", title_style))
        story.append(Paragraph("Advanced Healthcare AI System with Groq", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Patient information table
        patient_info = [
            ['Patient Name:', patient_data['name']],
            ['Email:', patient_data['email']],
            ['Phone:', patient_data.get('phone', 'Not provided')],
            ['Report Date:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ['Appointment ID:', appointment_details['appointment_id']],
            ['Urgency Level:', patient_data.get('urgency_level', 'Routine').title()]
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
        story.append(Paragraph("PRESENTING SYMPTOMS & CONCERNS", heading_style))
        for i, symptom in enumerate(patient_data['symptoms'], 1):
            story.append(Paragraph(f"• {symptom}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Enhanced medical analysis
        try:
            analysis_data = json.loads(medical_analysis) if isinstance(medical_analysis, str) else medical_analysis
            
            # Risk factors
            story.append(Paragraph("IDENTIFIED RISK FACTORS", heading_style))
            if analysis_data.get('risk_factors'):
                for risk in analysis_data['risk_factors']:
                    story.append(Paragraph(f"⚠️ {risk}", styles['Normal']))
            else:
                story.append(Paragraph("No specific risk factors identified from available information.", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Differential diagnosis
            if analysis_data.get('differential_diagnosis'):
                story.append(Paragraph("DIFFERENTIAL DIAGNOSIS CONSIDERATIONS", heading_style))
                for diagnosis in analysis_data['differential_diagnosis']:
                    story.append(Paragraph(f"• {diagnosis}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Medication alerts
            story.append(Paragraph("MEDICATION ALERTS & PRECAUTIONS", heading_style))
            if analysis_data.get('medication_alerts'):
                for alert in analysis_data['medication_alerts']:
                    story.append(Paragraph(f"🚨 {alert}", styles['Normal']))
            else:
                story.append(Paragraph("No specific medication alerts identified.", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Clinical correlation
            if analysis_data.get('clinical_correlation'):
                story.append(Paragraph("CLINICAL CORRELATION", heading_style))
                story.append(Paragraph(analysis_data['clinical_correlation'], styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Immediate care recommendations
            if analysis_data.get('immediate_care'):
                story.append(Paragraph("IMMEDIATE CARE RECOMMENDATIONS", heading_style))
                for care in analysis_data['immediate_care']:
                    story.append(Paragraph(f"✓ {care}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Clinical summary
            story.append(Paragraph("COMPREHENSIVE CLINICAL ASSESSMENT", heading_style))
            summary = analysis_data.get('summary', 'No summary available')
            story.append(Paragraph(summary, styles['Normal']))
            
        except Exception as e:
            story.append(Paragraph("MEDICAL ANALYSIS", heading_style))
            story.append(Paragraph(str(medical_analysis), styles['Normal']))
        
        story.append(Spacer(1, 25))
        
        # Enhanced appointment details
        story.append(Paragraph("SCHEDULED APPOINTMENT DETAILS", heading_style))
        appt_info = [
            ['Doctor:', appointment_details['doctor']],
            ['Specialty:', appointment_details.get('specialty', 'General Medicine')],
            ['Date:', appointment_details['date']],
            ['Time:', appointment_details['time']],
            ['Location:', appointment_details['location']],
            ['Contact:', appointment_details.get('doctor_email', 'Contact clinic directly')]
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
        
        # Pre-appointment instructions
        story.append(Spacer(1, 20))
        story.append(Paragraph("PRE-APPOINTMENT INSTRUCTIONS", heading_style))
        instructions = [
            "Arrive 15 minutes early with valid ID and insurance cards",
            "Bring all current medications and previous medical records",
            "Prepare a list of questions based on this medical analysis",
            "Fast for 8-12 hours if blood work is required",
            "Bring a family member if needed for support"
        ]
        for instruction in instructions:
            story.append(Paragraph(f"• {instruction}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, 
                                    textColor=colors.grey, alignment=1)
        story.append(Paragraph("AI-Generated Comprehensive Medical Report - Powered by Groq", footer_style))
        story.append(Paragraph("This report is generated by AI and should be reviewed by a qualified healthcare professional", footer_style))
        
        doc.build(story)
        return filename

class AutomatedEmailService:
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.email_address = EMAIL_ADDRESS
        self.email_password = EMAIL_PASSWORD
    
    def send_comprehensive_medical_email(self, patient_email: str, patient_name: str, 
                                       appointment_details: dict, pdf_path: str = None) -> bool:
        """Send comprehensive medical report via email automatically to the PATIENT'S email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = patient_email  # FIXED: Send to patient's email, not a fixed email
            msg['Subject'] = f"🏥 Comprehensive Medical Report & Appointment Confirmation - {patient_name}"
            
            # Enhanced email body
            email_body = f"""
Dear {patient_name},

Your comprehensive AI medical analysis has been completed successfully! 

🎯 APPOINTMENT CONFIRMED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👨‍⚕️ Doctor: {appointment_details['doctor']}
🏥 Specialty: {appointment_details.get('specialty', 'General Medicine')}
📅 Date: {appointment_details['date']}
⏰ Time: {appointment_details['time']}
📍 Location: {appointment_details['location']}
🆔 Appointment ID: {appointment_details['appointment_id']}
📧 Doctor's Email: {appointment_details.get('doctor_email', 'Contact clinic directly')}

📋 IMPORTANT INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Please review the attached comprehensive medical report before your appointment
✓ Arrive 15 minutes early with valid ID and insurance documentation
✓ Bring all current medications and previous medical records
✓ Prepare questions based on the AI medical analysis provided
✓ Fast for 8-12 hours if blood work may be required

🤖 AI ANALYSIS SUMMARY (Powered by Groq)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Our advanced AI medical system powered by Groq has analyzed your symptoms and medical history.
The detailed analysis, risk factors, and recommendations are included in the attached PDF report.

📞 CONTACT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• For appointment changes: {appointment_details.get('doctor_email', 'Contact clinic')}
• For emergencies: Call 108 (India Emergency Services)
• For technical support: healthcare.ai@support.com

🔒 PRIVACY & SECURITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This email and attachment contain confidential medical information.
Please keep this information secure and do not share with unauthorized persons.

Thank you for choosing our Advanced AI Healthcare System!

Best regards,
AI Healthcare Team (Powered by Groq)
Bengaluru Medical Network

---
This is an automated email from our AI Healthcare System.
Report generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
Patient Email: {patient_email}
"""
            
            msg.attach(MIMEText(email_body, 'plain'))
            
            # Attach PDF if available
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= comprehensive_medical_report_{patient_name.replace(" ", "_")}.pdf'
                )
                msg.attach(part)
            
            # Send email to patient's email address
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            print(f"✅ Email sent successfully to PATIENT: {patient_email}")
            return True
            
        except Exception as e:
            print(f"❌ Email sending failed: {str(e)}")
            return False

class EnhancedHealthcareCrewAI:
    def __init__(self):
        self.llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile", temperature=0.2)
        self.report_generator = EnhancedMedicalReportGenerator()
        self.email_service = AutomatedEmailService()
    
    def create_enhanced_medical_history_agent(self) -> Agent:
        return Agent(
            role="Senior Medical History Analyst",
            goal="Perform comprehensive analysis of patient medical history with advanced risk assessment",
            backstory="Expert medical AI with 20+ years equivalent experience in analyzing patient histories, "
                     "identifying complex risk factors, medication interactions, and providing detailed clinical insights "
                     "for healthcare providers. Specializes in comprehensive medical correlation analysis. Powered by Groq.",
            tools=[extract_comprehensive_medical_features],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_advanced_symptom_analyzer(self) -> Agent:
        return Agent(
            role="Advanced Clinical Symptom Diagnostician",
            goal="Analyze current symptoms with differential diagnosis and urgency classification",
            backstory="Highly advanced diagnostic AI that evaluates presenting symptoms using evidence-based medicine, "
                     "determines urgency levels, provides differential diagnosis considerations, and recommends "
                     "appropriate specialist referrals based on comprehensive clinical presentation analysis. Powered by Groq.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_intelligent_appointment_scheduler(self) -> Agent:
        return Agent(
            role="Intelligent Healthcare Appointment Coordinator",
            goal="Optimize appointment scheduling based on medical urgency and patient preferences",
            backstory="Advanced scheduling AI that considers symptom urgency, specialist availability, patient preferences, "
                     "and optimal care pathways to coordinate the most appropriate healthcare appointments. "
                     "Specializes in matching patients with the right specialists at the right time. Powered by Groq.",
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
            Perform comprehensive medical analysis for: {patient_data['name']}
            Email will be sent to: {patient_data['email']}
            
            PATIENT PROFILE:
            - Current Symptoms: {', '.join(patient_data['symptoms'])}
            - Medical History: {medical_history}
            - Urgency Level: {urgency}
            - Phone: {patient_data.get('phone', 'Not provided')}
            
            COMPREHENSIVE ANALYSIS REQUIRED:
            1. Advanced risk factor identification and stratification
            2. Medication alerts, interactions, and contraindications
            3. Differential diagnosis possibilities with likelihood assessment
            4. Clinical correlation between historical and current presentation
            5. Urgency assessment with triage recommendations
            6. Specialist recommendation with rationale
            7. Immediate care recommendations and red flag symptoms
            8. Comprehensive clinical summary with actionable insights
            
            Use the extract_comprehensive_medical_features tool for detailed analysis.
            Return structured, comprehensive medical assessment suitable for healthcare providers.
            """,
            agent=agent,
            expected_output="Comprehensive medical analysis in structured JSON format with all required components"
        )
    
    def create_advanced_symptom_assessment_task(self, agent: Agent, patient_data: dict, history_analysis: str) -> Task:
        return Task(
            description=f"""
            Advanced clinical symptom assessment for: {patient_data['name']}
            Patient Email: {patient_data['email']}
            
            CLINICAL PRESENTATION:
            - Presenting Symptoms: {', '.join(patient_data['symptoms'])}
            - Urgency Level: {patient_data.get('urgency_level', 'routine')}
            - Medical History Analysis: {history_analysis}
            
            ADVANCED ASSESSMENT REQUIREMENTS:
            1. Detailed symptom analysis with clinical significance
            2. Differential diagnosis with probability assessment
            3. Red flag symptom identification
            4. Urgency classification (EMERGENCY/URGENT/ROUTINE) with rationale
            5. Recommended specialist type with detailed justification
            6. Clinical correlation with medical history findings
            7. Immediate care recommendations and monitoring requirements
            8. Follow-up care pathway recommendations
            
            Provide evidence-based clinical assessment considering both current symptoms and historical context.
            """,
            agent=agent,
            expected_output="Detailed clinical assessment with urgency classification and specialist recommendations"
        )
    
    def create_intelligent_scheduling_task(self, agent: Agent, patient_data: dict, clinical_assessment: str) -> Task:
        preferred_date = patient_data.get('preferred_date', '')
        preferred_time = patient_data.get('preferred_time', '')
        urgency = patient_data.get('urgency_level', 'routine')
        
        return Task(
            description=f"""
            Intelligent appointment coordination for: {patient_data['name']}
            Patient Email: {patient_data['email']}
            
            SCHEDULING PARAMETERS:
            - Clinical Assessment: {clinical_assessment}
            - Patient Preferences: Date: {preferred_date}, Time: {preferred_time}
            - Urgency Level: {urgency}
            - Available Providers: {HEALTHCARE_PROVIDERS}
            
            SCHEDULING OPTIMIZATION:
            1. Match patient with most appropriate healthcare provider
            2. Consider urgency level for appointment timing
            3. Accommodate patient preferences when medically appropriate
            4. Optimize appointment timing based on clinical needs
            5. Provide pre-appointment preparation requirements
            6. Plan follow-up care coordination
            7. Consider provider availability and specialization match
            
            Use the schedule_optimal_appointment tool for intelligent scheduling.
            Provide comprehensive appointment coordination plan with rationale.
            """,
            agent=agent,
            expected_output="Detailed appointment coordination with provider matching and optimal timing"
        )
    
    def process_patient_with_auto_email(self, patient_data: dict) -> dict:
        """Enhanced patient processing with automatic email delivery to PATIENT'S email"""
        print(f"\n🚀 Starting comprehensive medical processing for: {patient_data['name']}")
        print(f"📧 Email will be sent to: {patient_data['email']}")
        print("=" * 70)
        
        try:
            # Create enhanced specialized agents
            print("🤖 Initializing advanced medical AI agents with Groq...")
            history_agent = self.create_enhanced_medical_history_agent()
            symptom_agent = self.create_advanced_symptom_analyzer()
            scheduler_agent = self.create_intelligent_appointment_scheduler()
            
            # Step 1: Comprehensive Medical History Analysis
            print("\n📋 STEP 1: Comprehensive Medical History Analysis")
            print("🔍 Analyzing patient history, risk factors, and medical correlations...")
            history_task = self.create_comprehensive_medical_analysis_task(history_agent, patient_data)
            history_crew = Crew(agents=[history_agent], tasks=[history_task], process=Process.sequential)
            history_analysis = history_crew.kickoff()
            print("✅ Medical history analysis completed")
            
            # Step 2: Advanced Clinical Symptom Assessment
            print("\n🩺 STEP 2: Advanced Clinical Symptom Assessment")
            print("🔬 Evaluating symptoms, differential diagnosis, and urgency...")
            symptom_task = self.create_advanced_symptom_assessment_task(symptom_agent, patient_data, str(history_analysis))
            symptom_crew = Crew(agents=[symptom_agent], tasks=[symptom_task], process=Process.sequential)
            clinical_assessment = symptom_crew.kickoff()
            print("✅ Clinical symptom assessment completed")
            
            # Step 3: Intelligent Appointment Coordination
            print("\n📅 STEP 3: Intelligent Appointment Coordination")
            print("🎯 Matching with optimal healthcare provider and scheduling...")
            scheduling_task = self.create_intelligent_scheduling_task(scheduler_agent, patient_data, str(clinical_assessment))
            scheduling_crew = Crew(agents=[scheduler_agent], tasks=[scheduling_task], process=Process.sequential)
            appointment_coordination = scheduling_crew.kickoff()
            print("✅ Appointment coordination completed")
            
            # Step 4: Generate Comprehensive PDF Report
            print("\n📄 STEP 4: Generating Comprehensive Medical Report")
            print("📝 Creating detailed PDF report with all analysis results...")
            
            # Extract appointment details from coordination result
            coordination_text = str(appointment_coordination).lower()
            urgency = patient_data.get('urgency_level', 'routine')
            
            # Use the scheduling tool to get structured appointment details
            appointment_details = schedule_optimal_appointment(
                patient_data['symptoms'], 
                urgency, 
                patient_data.get('preferred_date', ''),
                patient_data.get('preferred_time', '')
            )
            
            # Parse the tool result if it's a string
            if isinstance(appointment_details, str):
                try:
                    appointment_details = json.loads(appointment_details)
                except:
                    # Fallback appointment details
                    appointment_details = {
                        'doctor': 'Dr. Amit Singh',
                        'specialty': 'Internal Medicine',
                        'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                        'time': '10:00 AM',
                        'location': 'City General Hospital, Bengaluru',
                        'doctor_email': 'amit.singh@generalhospital.com',
                        'appointment_id': f"APPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    }
            
            pdf_path = self.report_generator.generate_comprehensive_pdf_report(
                patient_data, str(history_analysis), appointment_details
            )
            print(f"✅ Comprehensive PDF report generated: {pdf_path}")
            
            # Step 5: Automatic Email Delivery to PATIENT'S EMAIL
            print("\n📧 STEP 5: Automatic Email Delivery")
            print(f"📮 Sending comprehensive medical report to PATIENT'S email: {patient_data['email']}")
            
            email_sent = self.email_service.send_comprehensive_medical_email(
                patient_data['email'],  # FIXED: Use patient's email from the form
                patient_data['name'],
                appointment_details,
                pdf_path
            )
            
            if email_sent:
                print(f"✅ Email delivered successfully to: {patient_data['email']}")
                email_status = f"Email sent successfully to {patient_data['email']} with comprehensive medical report"
            else:
                print("❌ Email delivery failed")
                email_status = "Email delivery failed - please check email configuration"
            
            print("\n🎉 COMPREHENSIVE MEDICAL PROCESSING COMPLETED!")
            print(f"📊 All AI agents utilized successfully with Groq")
            print(f"📄 Medical report: {pdf_path}")
            print(f"📧 Email status: {email_status}")
            
            return {
                'success': True,
                'patient_info': patient_data,
                'medical_history_analysis': str(history_analysis),
                'clinical_assessment': str(clinical_assessment),
                'appointment_coordination': str(appointment_coordination),
                'appointment_details': appointment_details,
                'pdf_report_path': pdf_path,
                'email_sent': email_sent,
                'email_status': email_status,
                'urgency': urgency,
                'processing_summary': 'All CrewAI agents utilized with Groq: Medical History Analyst, Symptom Diagnostician, Appointment Coordinator'
            }
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}

# Test the enhanced system
if __name__ == "__main__":
    print("🚀 Testing Enhanced Healthcare CrewAI System with Groq and Fixed Email")
    
    # Test patient data
    test_patient = {
        'name': 'John Doe',
        'email': 'john.doe@example.com',  # This email will receive the report
        'phone': '+91-9876543210',
        'symptoms': ['chest pain', 'shortness of breath', 'fatigue'],
        'medical_history': 'Hypertension, family history of heart disease',
        'urgency_level': 'urgent',
        'preferred_date': '2024-01-15',
        'preferred_time': '10:00 AM'
    }
    
    healthcare_system = EnhancedHealthcareCrewAI()
    results = healthcare_system.process_patient_with_auto_email(test_patient)
    
    if results['success']:
        print("\n✅ Test completed successfully!")
        print(f"📧 Email sent to patient: {results['email_sent']}")
    else:
        print(f"\n❌ Test failed: {results['error']}")
