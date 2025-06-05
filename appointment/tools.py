from crewai.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import json
import base64
import tempfile
import webbrowser
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

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
        "email": "emergency@hospital.com", 
        "location": "Emergency Department, City Hospital", 
        "specializations": ["emergency", "critical", "urgent care"]
    }
}

@tool
def parse_user_input(user_description: str) -> dict:
    """Parse natural language user input to extract medical information"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.2, max_tokens=500)
    
    prompt = f"""
    Parse this patient's natural language description and extract structured medical information:
    
    Patient Description: "{user_description}"
    
    Extract and return JSON format:
    {{
        "symptoms": ["symptom1", "symptom2", ...],
        "urgency_level": "emergency/urgent/routine",
        "medical_specialty_needed": "cardiology/neurology/internal_medicine/emergency",
        "emergency_keywords": ["keyword1", "keyword2", ...],
        "duration": "how long symptoms present",
        "severity": "mild/moderate/severe",
        "context": "additional relevant context"
    }}
    
    Emergency keywords include: chest pain, can't breathe, unconscious, severe bleeding, stroke, heart attack, etc.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "symptoms": ["Unable to parse symptoms"],
            "urgency_level": "routine",
            "medical_specialty_needed": "internal_medicine",
            "emergency_keywords": [],
            "duration": "unknown",
            "severity": "unknown",
            "context": f"Parsing error: {str(e)}"
        })

@tool
def determine_routing_strategy(parsed_input: str) -> dict:
    """Determine which agents should be activated based on parsed input"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1, max_tokens=300)
    
    prompt = f"""
    Based on this parsed medical input, determine the routing strategy:
    
    Parsed Input: {parsed_input}
    
    Available Agents:
    - emergency_alert_agent: For life-threatening conditions
    - triage_agent: For initial assessment
    - symptom_analyzer: For symptom analysis
    - medical_history_agent: For history analysis
    - appointment_scheduler: For scheduling
    - general_practitioner_agent: For routine care
    
    Return JSON:
    {{
        "primary_agents": ["agent1", "agent2"],
        "secondary_agents": ["agent3"],
        "execution_order": ["first", "second", "third"],
        "emergency_protocol": true/false,
        "reasoning": "why these agents were selected"
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "primary_agents": ["triage_agent", "symptom_analyzer"],
            "secondary_agents": ["appointment_scheduler"],
            "execution_order": ["triage_agent", "symptom_analyzer", "appointment_scheduler"],
            "emergency_protocol": False,
            "reasoning": f"Default routing due to error: {str(e)}"
        })

@tool
def extract_medical_features(medical_history: str) -> dict:
    """Extract medical features from patient history using LLM"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
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
        if 'symptoms' in patient_data:
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
                story.append(Paragraph("No specific risk factors identified.", styles['Normal']))
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
        story.append(Paragraph("AI-Generated Medical Report", footer_style))
        
        doc.build(story)
        return filename

def create_web_email_interface(patient_email: str, subject: str, body: str, pdf_path: str = None) -> str:
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        webbrowser.open(f'file://{f.name}')
    
    return html_content
