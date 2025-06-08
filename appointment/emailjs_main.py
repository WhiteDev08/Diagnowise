from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import os
import json
import uuid
from datetime import datetime, timedelta

# Import Groq crew modules
from emailjs_crew import EnhancedHealthcareCrewAI, HEALTHCARE_PROVIDERS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Groq Healthcare AI API",
    description="Advanced Healthcare AI System with Groq Integration and Fixed Email Delivery",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq healthcare system
healthcare_system = EnhancedHealthcareCrewAI()

# Enhanced in-memory storage
appointments_db = {}
patients_db = {}
reports_db = {}
email_logs = {}

# Groq-optimized Pydantic models
class GroqPatientData(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history reported."
    appointment_type: Optional[str] = "consultation"
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class GroqMedicalAnalysisRequest(BaseModel):
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history."
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

# Groq API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Groq Healthcare AI API is running",
        "version": "3.0.0",
        "llm_provider": "Groq (llama-3.1-70b-versatile)",
        "email_fix": "Fixed - Emails sent to patient's email address",
        "features": [
            "Groq LLM Integration",
            "Fixed Email Delivery to Patient",
            "Comprehensive AI Medical Analysis",
            "Date/Time Preferences",
            "All CrewAI Agents Utilized",
            "Enhanced PDF Reports"
        ],
        "endpoints": [
            "/providers",
            "/appointments",
            "/process-patient-groq",
            "/medical-analysis-groq",
            "/email-logs"
        ]
    }

@app.get("/providers")
async def get_groq_providers(specialty: Optional[str] = None):
    """Get healthcare providers optimized for Groq delivery"""
    try:
        providers_list = []
        for spec, provider in HEALTHCARE_PROVIDERS.items():
            if specialty and specialty.lower() not in spec.lower():
                continue
                
            providers_list.append({
                "id": spec,
                "name": provider["name"],
                "specialty": spec.replace("_", " ").title(),
                "email": provider["email"],
                "location": provider["location"],
                "specializations": provider["specializations"],
                "available_slots": provider.get("available_slots", []),
                "rating": 4.8,
                "nextAvailable": "Next Week",
                "experience": "15+ years",
                "groq_enabled": True
            })
        
        return {"providers": providers_list, "total": len(providers_list), "groq_ready": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-patient-groq")
async def process_patient_with_groq(patient_data: GroqPatientData, background_tasks: BackgroundTasks):
    """Enhanced patient processing with Groq and fixed email delivery"""
    try:
        patient_id = str(uuid.uuid4())
        
        # Convert to dict for processing
        patient_dict = {
            "name": patient_data.name,
            "email": patient_data.email,  # This will be used for email delivery
            "phone": patient_data.phone or "Not provided",
            "symptoms": patient_data.symptoms,
            "medical_history": patient_data.medical_history,
            "preferred_date": patient_data.preferred_date,
            "preferred_time": patient_data.preferred_time,
            "urgency_level": patient_data.urgency_level
        }
        
        # Store patient data
        patients_db[patient_id] = patient_dict
        
        print(f"🚀 Groq processing started for: {patient_data.name}")
        print(f"📧 Email will be sent to: {patient_data.email}")
        
        # Process in background with Groq
        background_tasks.add_task(process_patient_groq_background, patient_id, patient_dict)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "patient_email": patient_data.email,
            "message": f"Enhanced AI processing started with Groq. Medical report will be sent to {patient_data.email}",
            "status": "processing",
            "llm_provider": "Groq (llama-3.1-70b-versatile)",
            "email_fix": "Fixed - Will send to patient's email",
            "features_used": [
                "Medical History Analyst Agent (Groq)",
                "Symptom Diagnostician Agent (Groq)", 
                "Appointment Coordinator Agent (Groq)",
                "Fixed Email Delivery to Patient",
                "Comprehensive PDF Report"
            ]
        }
        
    except Exception as e:
        print(f"❌ Groq processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groq processing failed: {str(e)}")

async def process_patient_groq_background(patient_id: str, patient_data: dict):
    """Enhanced background processing with Groq and fixed email"""
    try:
        print(f"🤖 Starting Groq background processing for: {patient_id}")
        print(f"📧 Target email: {patient_data['email']}")
        
        # Process through Groq crew system
        results = healthcare_system.process_patient_with_auto_email(patient_data)
        
        # Store comprehensive results
        if results['success']:
            appointments_db[patient_id] = {
                "patient_id": patient_id,
                "appointment_details": results['appointment_details'],
                "medical_analysis": results['medical_history_analysis'],
                "clinical_assessment": results['clinical_assessment'],
                "appointment_coordination": results['appointment_coordination'],
                "pdf_report_path": results.get('pdf_report_path'),
                "urgency": results.get('urgency', 'routine'),
                "status": "confirmed",
                "email_sent": results.get('email_sent', False),
                "email_status": results.get('email_status', ''),
                "patient_email": patient_data['email'],  # Store patient email
                "processing_summary": results.get('processing_summary', ''),
                "llm_provider": "Groq",
                "created_at": datetime.now().isoformat()
            }
            
            # Store report info
            reports_db[patient_id] = {
                "patient_id": patient_id,
                "report_path": results.get('pdf_report_path'),
                "generated_at": datetime.now().isoformat(),
                "email_delivered": results.get('email_sent', False),
                "patient_email": patient_data['email']
            }
            
            # Log email delivery
            email_logs[patient_id] = {
                "patient_id": patient_id,
                "patient_email": patient_data['email'],
                "email_sent": results.get('email_sent', False),
                "email_status": results.get('email_status', ''),
                "llm_provider": "Groq",
                "timestamp": datetime.now().isoformat()
            }
            
        print(f"✅ Groq background processing completed for: {patient_id}")
        
    except Exception as e:
        print(f"❌ Groq background processing failed: {str(e)}")
        # Store error info
        appointments_db[patient_id] = {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e),
            "patient_email": patient_data['email'],
            "llm_provider": "Groq",
            "created_at": datetime.now().isoformat()
        }

@app.post("/medical-analysis-groq")
async def get_groq_medical_analysis(request: GroqMedicalAnalysisRequest):
    """Enhanced medical analysis with Groq"""
    try:
        # Create Groq-compatible temporary patient data
        temp_patient = {
            "name": "Analysis Request",
            "email": "temp@example.com",
            "symptoms": request.symptoms,
            "medical_history": request.medical_history,
            "preferred_date": request.preferred_date,
            "preferred_time": request.preferred_time,
            "urgency_level": request.urgency_level
        }
        
        # Run Groq-optimized analysis
        history_agent = healthcare_system.create_enhanced_medical_history_agent()
        history_task = healthcare_system.create_comprehensive_medical_analysis_task(history_agent, temp_patient)
        
        from crewai import Crew, Process
        history_crew = Crew(agents=[history_agent], tasks=[history_task], process=Process.sequential)
        analysis_result = history_crew.kickoff()
        
        # Determine recommended specialty and provider
        symptom_text = " ".join(request.symptoms).lower()
        best_specialty = "internal_medicine"
        
        for specialty, provider in HEALTHCARE_PROVIDERS.items():
            if any(spec in symptom_text for spec in provider['specializations']):
                best_specialty = specialty
                break
        
        recommended_provider = HEALTHCARE_PROVIDERS[best_specialty]
        
        # Groq-compatible scheduling note
        scheduling_note = ""
        if request.preferred_date or request.preferred_time:
            scheduling_note = f"Groq analysis includes preferences: Date: {request.preferred_date or 'Flexible'}, Time: {request.preferred_time or 'Flexible'}"
        
        return {
            "success": True,
            "analysis": str(analysis_result),
            "recommended_specialty": best_specialty.replace("_", " ").title(),
            "recommended_provider": recommended_provider,
            "urgency": request.urgency_level,
            "scheduling_preferences": scheduling_note,
            "available_slots": recommended_provider.get("available_slots", []),
            "llm_provider": "Groq (llama-3.1-70b-versatile)",
            "groq_compatible": True,
            "analysis_features": [
                "Risk factor assessment (Groq)",
                "Medication alerts (Groq)",
                "Differential diagnosis (Groq)",
                "Clinical correlation (Groq)",
                "Urgency classification (Groq)",
                "Fixed email delivery"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/appointments")
async def get_all_groq_appointments():
    """Get all appointments with Groq details"""
    try:
        appointments_list = []
        for patient_id, appt in appointments_db.items():
            patient = patients_db.get(patient_id, {})
            appointment_details = appt.get('appointment_details', {})
            
            appointments_list.append({
                "id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "patient_email": appt.get('patient_email', 'Unknown'),
                "provider": appointment_details.get('doctor', 'TBD'),
                "specialty": appointment_details.get('specialty', 'General'),
                "date": appointment_details.get('date', 'TBD'),
                "time": appointment_details.get('time', 'TBD'),
                "type": "Groq AI Consultation",
                "location": appointment_details.get('location', 'TBD'),
                "status": appt.get('status', 'pending'),
                "urgency": appt.get('urgency', 'routine'),
                "email_sent": appt.get('email_sent', False),
                "email_status": appt.get('email_status', ''),
                "llm_provider": appt.get('llm_provider', 'Groq'),
                "processing_summary": appt.get('processing_summary', ''),
                "created_at": appt.get('created_at', '')
            })
        
        return {
            "appointments": appointments_list,
            "total": len(appointments_list),
            "groq_processed_count": sum(1 for a in appointments_list if a['llm_provider'] == 'Groq'),
            "email_delivery_rate": sum(1 for a in appointments_list if a['email_sent']) / max(len(appointments_list), 1) * 100,
            "llm_provider": "Groq"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/email-logs")
async def get_groq_email_logs():
    """Get email delivery logs with Groq processing"""
    try:
        logs_list = []
        for patient_id, log in email_logs.items():
            patient = patients_db.get(patient_id, {})
            logs_list.append({
                "patient_id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "patient_email": log['patient_email'],
                "email_sent": log['email_sent'],
                "email_status": log['email_status'],
                "llm_provider": log.get('llm_provider', 'Groq'),
                "timestamp": log['timestamp']
            })
        
        return {
            "email_logs": logs_list,
            "total_emails": len(logs_list),
            "successful_deliveries": sum(1 for log in logs_list if log['email_sent']),
            "delivery_rate": sum(1 for log in logs_list if log['email_sent']) / max(len(logs_list), 1) * 100,
            "groq_processed": sum(1 for log in logs_list if log['llm_provider'] == 'Groq'),
            "llm_provider": "Groq",
            "email_fix": "Fixed - All emails sent to patient addresses"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-slots")
async def get_groq_available_slots(date: str, provider: Optional[str] = None):
    """Get available time slots with Groq compatibility"""
    try:
        # Get provider-specific slots if specified
        if provider:
            for specialty, prov_data in HEALTHCARE_PROVIDERS.items():
                if prov_data["name"] == provider:
                    base_slots = prov_data.get("available_slots", [
                        "9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"
                    ])
                    break
            else:
                base_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
        else:
            base_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
        
        # Filter out booked slots
        available_slots = base_slots.copy()
        
        for appt in appointments_db.values():
            appt_details = appt.get("appointment_details", {})
            if appt_details.get("date") == date and appt.get("status") == "confirmed":
                booked_time = appt_details.get("time")
                if booked_time in available_slots:
                    available_slots.remove(booked_time)
        
        return {
            "success": True,
            "date": date,
            "provider": provider,
            "available_slots": available_slots,
            "total_slots": len(available_slots),
            "groq_compatible": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{patient_id}")
async def get_groq_medical_report(patient_id: str):
    """Download Groq-generated medical report PDF"""
    try:
        if patient_id not in reports_db:
            raise HTTPException(status_code=404, detail="Groq medical report not found")
        
        report_info = reports_db[patient_id]
        pdf_path = report_info["report_path"]
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=pdf_path,
            filename=f"groq_medical_report_{patient_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/health")
async def groq_health_check():
    """Groq system health check"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "llm_provider": "Groq (llama-3.1-70b-versatile)",
        "email_fix": "Fixed - Emails sent to patient addresses",
        "timestamp": datetime.now().isoformat(),
        "patients_count": len(patients_db),
        "appointments_count": len(appointments_db),
        "reports_count": len(reports_db),
        "emails_sent": len(email_logs),
        "email_delivery_rate": sum(1 for log in email_logs.values() if log['email_sent']) / max(len(email_logs), 1) * 100,
        "groq_processed": sum(1 for log in email_logs.values() if log.get('llm_provider') == 'Groq'),
        "features": [
            "Groq LLM Integration",
            "Fixed Email Delivery to Patient", 
            "Date/Time Preferences",
            "Comprehensive PDF Reports",
            "All AI Agents Utilized"
        ]
    }

@app.post("/system/reset")
async def reset_groq_system():
    """Reset all Groq system data"""
    global appointments_db, patients_db, reports_db, email_logs
    appointments_db.clear()
    patients_db.clear()
    reports_db.clear()
    email_logs.clear()
    
    return {
        "success": True,
        "message": "Groq system data reset successfully",
        "llm_provider": "Groq",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Groq Healthcare AI FastAPI Server...")
    print("✨ New Features:")
    print("   - Groq LLM Integration (llama-3.1-70b-versatile)")
    print("   - FIXED: Email delivery to patient's email address")
    print("   - Enhanced AI Medical Analysis")
    print("   - Date/Time Preferences")
    print("   - All CrewAI Agents Utilized")
    print("   - Comprehensive PDF Reports")
    print("\n📋 Groq Endpoints:")
    print("   - POST /process-patient-groq")
    print("   - POST /medical-analysis-groq") 
    print("   - GET  /email-logs")
    print("   - GET  /providers (Groq ready)")
    print("   - GET  /appointments (Groq enhanced)")
    print("\n🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n🔧 Required Environment Variables:")
    print("   - GROQ_API_KEY (your Groq API key)")
    print("   - EMAIL_ADDRESS (your Gmail address)")
    print("   - EMAIL_APP_PASSWORD (Gmail app password)")
    
    uvicorn.run(
        "groq_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
