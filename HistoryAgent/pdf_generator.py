# pdf_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import json
from datetime import datetime
import os

def generate_medical_report_pdf(analysis_result, patient_name="Patient", output_dir="reports"):
    """Generate a formatted PDF report from medical analysis results."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle CrewOutput object or string
    if hasattr(analysis_result, 'raw'):
        # It's a CrewOutput object, get the raw content
        result_text = str(analysis_result.raw)
    else:
        # It's already a string
        result_text = str(analysis_result)
    
    # Parse JSON from the text
    try:
        # First try direct JSON parsing
        data = json.loads(result_text)
    except:
        try:
            # Try to extract JSON from string if it contains other text
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"error": "Could not parse analysis result", "raw_output": result_text}
        except:
            data = {"error": "Could not parse analysis result", "raw_output": result_text}
    
    # Debug print to see what data we're working with
    print(f"DEBUG - Final parsed data: {data}")
    print(f"DEBUG - Risk factors found: {data.get('risk_factors', [])}")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/medical_analysis_{patient_name.replace(' ', '_')}_{timestamp}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkred,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=5,
        backColor=colors.lightgrey
    )
    
    # Title
    story.append(Paragraph("MEDICAL HISTORY ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Patient info and date
    info_data = [
        ['Patient:', patient_name],
        ['Report Date:', datetime.now().strftime("%B %d, %Y")],
        ['Report Time:', datetime.now().strftime("%I:%M %p")]
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    # Risk Factors Section
    story.append(Paragraph("IDENTIFIED RISK FACTORS", heading_style))
    story.append(Spacer(1, 10))
    
    if 'risk_factors' in data and data['risk_factors']:
        risk_data = []
        for i, risk in enumerate(data['risk_factors'], 1):
            risk_data.append([f"{i}.", risk])
        
        risk_table = Table(risk_data, colWidths=[0.5*inch, 5.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(risk_table)
    else:
        story.append(Paragraph("No specific risk factors identified.", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Medication Alerts Section
    story.append(Paragraph("MEDICATION ALERTS", heading_style))
    story.append(Spacer(1, 10))
    
    if 'medication_alerts' in data and data['medication_alerts']:
        alert_data = []
        for i, alert in enumerate(data['medication_alerts'], 1):
            alert_data.append([f"{i}.", alert])
        
        alert_table = Table(alert_data, colWidths=[0.5*inch, 5.5*inch])
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightcoral),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(alert_table)
    else:
        story.append(Paragraph("No medication alerts identified.", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Summary Section
    story.append(Paragraph("CLINICAL SUMMARY", heading_style))
    story.append(Spacer(1, 10))
    
    if 'summary' in data and data['summary']:
        summary_text = data['summary']
        story.append(Paragraph(summary_text, styles['Normal']))
    else:
        story.append(Paragraph("No summary available.", styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph("This report is generated by AI Medical History Analyzer", footer_style))
    story.append(Paragraph("For clinical use only - Please consult with healthcare professionals", footer_style))
    
    # Build PDF
    doc.build(story)
    return filename