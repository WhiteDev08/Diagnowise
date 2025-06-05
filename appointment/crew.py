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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    }
}

@tool
def extract_medical_features(medical_history: str) -> dict:
    """Extract medical features from patient history using LLM"""
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
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        return json.dumps(result, indent=2)
    except:
        return json.dumps({"risk_factors": [], "medication_alerts": [], "summary": "Analysis unavailable"})

class MedicalReportGenerator:
    @staticmethod
    def generate_pdf_report(patient_data: dict, medical_analysis: str, appointment_details: dict) -> str:
        """Generate comprehensive medical PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_report_{patient_data['name'].replace(' ', '_')}_{timestamp}.pdf"
        
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
            ['Email:', patient_data['email']],
            ['Report Date:', datetime.now().strftime("%B %d, %Y")],
            ['Appointment ID:', appointment_details['appointment_id']]
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
        story.append(Paragraph("PRESENTING SYMPTOMS", heading_style))
        for i, symptom in enumerate(patient_data['symptoms'], 1):
            story.append(Paragraph(f"{i}. {symptom}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Medical analysis
        try:
            analysis_data = json.loads(medical_analysis) if isinstance(medical_analysis, str) else medical_analysis
            
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
            summary = analysis_data.get('summary', 'No summary available')
            story.append(Paragraph(summary, styles['Normal']))
            
        except:
            story.append(Paragraph("MEDICAL ANALYSIS", heading_style))
            story.append(Paragraph(str(medical_analysis), styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Appointment details
        story.append(Paragraph("SCHEDULED APPOINTMENT", heading_style))
        appt_info = [
            ['Doctor:', appointment_details['doctor']],
            ['Specialty:', appointment_details['specialty']],
            ['Date:', appointment_details['date']],
            ['Time:', appointment_details['time']],
            ['Location:', appointment_details['location']]
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
        return filename

class HealthcareCrewAI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.3)
        self.report_generator = MedicalReportGenerator()
    
    def create_medical_history_agent(self) -> Agent:
        return Agent(
            role="Medical History Analyzer",
            goal="Analyze patient medical history and extract comprehensive medical insights",
            backstory="Expert medical AI that analyzes patient histories, identifies risk factors, "
                     "medication interactions, and provides clinical summaries for healthcare providers.",
            tools=[extract_medical_features],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_symptom_analyzer(self) -> Agent:
        return Agent(
            role="Clinical Symptom Analyzer",
            goal="Analyze current symptoms and provide medical assessment with urgency classification",
            backstory="Advanced diagnostic AI that evaluates presenting symptoms, determines urgency levels, "
                     "and recommends appropriate specialist referrals based on clinical presentation.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_appointment_scheduler(self) -> Agent:
        return Agent(
            role="Healthcare Appointment Coordinator",
            goal="Match patients with appropriate specialists and optimize appointment scheduling",
            backstory="Intelligent scheduling system that considers symptom urgency, specialist availability, "
                     "and patient needs to coordinate optimal healthcare appointments.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_medical_history_task(self, agent: Agent, patient_data: dict) -> Task:
        medical_history = patient_data.get('medical_history', 'No previous medical history provided.')
        
        return Task(
            description=f"""
            Analyze comprehensive medical profile for: {patient_data['name']}
            
            Current Symptoms: {', '.join(patient_data['symptoms'])}
            Medical History: {medical_history}
            
            Provide detailed analysis including:
            1. Risk factor identification and assessment
            2. Medication alerts and contraindications
            3. Clinical correlation between history and current symptoms
            4. Comprehensive medical summary with recommendations
            
            Return structured JSON format for integration.
            """,
            agent=agent,
            expected_output="Comprehensive medical analysis in JSON format with risk factors, alerts, and clinical summary"
        )
    
    def create_symptom_analysis_task(self, agent: Agent, patient_data: dict, history_analysis: str) -> Task:
        return Task(
            description=f"""
            Clinical symptom assessment for: {patient_data['name']}
            
            Presenting Symptoms: {', '.join(patient_data['symptoms'])}
            Medical History Analysis: {history_analysis}
            
            Provide:
            1. Differential diagnosis considerations
            2. Urgency classification (EMERGENCY/URGENT/ROUTINE)
            3. Recommended specialist type and rationale
            4. Clinical correlation with medical history
            5. Immediate care recommendations
            
            Consider both current symptoms and historical medical context.
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
            scheduler_agent = self.create_appointment_scheduler()
            
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
    """Enhanced main execution with comprehensive medical processing"""
    if not openai_api_key:
        print("❌ ERROR: OpenAI API key not found in .env file")
        return
    
    print("🚀 Starting Advanced Healthcare AI System...")
    
    try:
        healthcare_system = HealthcareCrewAI()
        
        while True:
            patient_data = get_enhanced_patient_input()
            results = healthcare_system.process_patient(patient_data)
            
            if results['success']:
                print(f"\n🎉 Advanced medical processing completed successfully!")
                print(f"📊 Comprehensive analysis generated with PDF report")
            else:
                print(f"❌ Processing failed: {results['error']}")
            
            continue_choice = input("\n🔄 Process another patient? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("👋 Thank you for using Advanced Healthcare AI System!")
                break
                
    except Exception as e:
        print(f"❌ System error: {str(e)}")

if __name__ == "__main__":
    main()