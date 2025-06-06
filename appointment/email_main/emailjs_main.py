from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import os
import json
import uuid
from datetime import datetime, timedelta

# Import EmailJS crew modules
from emailjs_crew import EmailJSHealthcareCrewAI, HEALTHCARE_PROVIDERS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="EmailJS Healthcare AI API",
    description="Advanced Healthcare AI System with EmailJS Integration",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EmailJS healthcare system
healthcare_system = EmailJSHealthcareCrewAI()

# Enhanced in-memory storage
appointments_db = {}
patients_db = {}
reports_db = {}
emailjs_logs = {}

# EmailJS-optimized Pydantic models
class EmailJSPatientData(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history reported."
    appointment_type: Optional[str] = "consultation"
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class EmailJSMedicalAnalysisRequest(BaseModel):
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history."
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

# EmailJS API Endpoints

@app.get("/")
async def root():
    return {
        "message": "EmailJS Healthcare AI API is running",
        "version": "2.1.0",
        "email_service": "EmailJS Integration",
        "features": [
            "EmailJS Automatic Email Delivery",
            "Comprehensive AI Medical Analysis",
            "Date/Time Preferences",
            "All CrewAI Agents Utilized",
            "Enhanced PDF Reports"
        ],
        "endpoints": [
            "/providers",
            "/appointments",
            "/process-patient-enhanced",
            "/medical-analysis-enhanced",
            "/emailjs-logs"
        ]
    }

@app.get("/providers")
async def get_emailjs_providers(specialty: Optional[str] = None):
    """Get healthcare providers optimized for EmailJS delivery"""
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
                "emailjs_enabled": True
            })
        
        return {"providers": providers_list, "total": len(providers_list), "emailjs_ready": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-patient-enhanced")
async def process_patient_with_emailjs(patient_data: EmailJSPatientData, background_tasks: BackgroundTasks):
    """Enhanced patient processing with EmailJS delivery"""
    try:
        patient_id = str(uuid.uuid4())
        
        # Convert to dict for processing
        patient_dict = {
            "name": patient_data.name,
            "email": patient_data.email,
            "phone": patient_data.phone or "Not provided",
            "symptoms": patient_data.symptoms,
            "medical_history": patient_data.medical_history,
            "preferred_date": patient_data.preferred_date,
            "preferred_time": patient_data.preferred_time,
            "urgency_level": patient_data.urgency_level
        }
        
        # Store patient data
        patients_db[patient_id] = patient_dict
        
        print(f"🚀 EmailJS processing started for: {patient_data.name}")
        
        # Process in background with EmailJS
        background_tasks.add_task(process_patient_emailjs_background, patient_id, patient_dict)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "message": "Enhanced AI processing started. Medical report will be sent via EmailJS automatically.",
            "status": "processing",
            "email_service": "EmailJS",
            "features_used": [
                "Medical History Analyst Agent",
                "Symptom Diagnostician Agent", 
                "Appointment Coordinator Agent",
                "EmailJS Automatic Delivery",
                "Comprehensive PDF Report"
            ]
        }
        
    except Exception as e:
        print(f"❌ EmailJS processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EmailJS processing failed: {str(e)}")

async def process_patient_emailjs_background(patient_id: str, patient_data: dict):
    """Enhanced background processing with EmailJS preparation"""
    try:
        print(f"🤖 Starting EmailJS background processing for: {patient_id}")
        
        # Process through EmailJS crew system
        results = healthcare_system.process_patient_for_emailjs(patient_data)
        
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
                "emailjs_data": results.get('emailjs_data', {}),
                "emailjs_ready": True,
                "processing_summary": results.get('processing_summary', ''),
                "created_at": datetime.now().isoformat()
            }
            
            # Store report info
            reports_db[patient_id] = {
                "patient_id": patient_id,
                "report_path": results.get('pdf_report_path'),
                "generated_at": datetime.now().isoformat(),
                "emailjs_prepared": True
            }
            
            # Log EmailJS preparation
            emailjs_logs[patient_id] = {
                "patient_id": patient_id,
                "patient_email": patient_data['email'],
                "emailjs_data_prepared": True,
                "emailjs_data": results.get('emailjs_data', {}),
                "timestamp": datetime.now().isoformat()
            }
            
        print(f"✅ EmailJS background processing completed for: {patient_id}")
        
    except Exception as e:
        print(f"❌ EmailJS background processing failed: {str(e)}")
        # Store error info
        appointments_db[patient_id] = {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.now().isoformat()
        }

@app.post("/medical-analysis-enhanced")
async def get_emailjs_medical_analysis(request: EmailJSMedicalAnalysisRequest):
    """Enhanced medical analysis with EmailJS compatibility"""
    try:
        # Create EmailJS-compatible temporary patient data
        temp_patient = {
            "name": "Analysis Request",
            "email": "temp@example.com",
            "symptoms": request.symptoms,
            "medical_history": request.medical_history,
            "preferred_date": request.preferred_date,
            "preferred_time": request.preferred_time,
            "urgency_level": request.urgency_level
        }
        
        # Run EmailJS-optimized analysis
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
        
        # EmailJS-compatible scheduling note
        scheduling_note = ""
        if request.preferred_date or request.preferred_time:
            scheduling_note = f"EmailJS will include preferences: Date: {request.preferred_date or 'Flexible'}, Time: {request.preferred_time or 'Flexible'}"
        
        return {
            "success": True,
            "analysis": str(analysis_result),
            "recommended_specialty": best_specialty.replace("_", " ").title(),
            "recommended_provider": recommended_provider,
            "urgency": request.urgency_level,
            "scheduling_preferences": scheduling_note,
            "available_slots": recommended_provider.get("available_slots", []),
            "emailjs_compatible": True,
            "analysis_features": [
                "Risk factor assessment",
                "Medication alerts",
                "Differential diagnosis",
                "Clinical correlation",
                "Urgency classification",
                "EmailJS optimization"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/appointments")
async def get_all_emailjs_appointments():
    """Get all appointments with EmailJS details"""
    try:
        appointments_list = []
        for patient_id, appt in appointments_db.items():
            patient = patients_db.get(patient_id, {})
            appointment_details = appt.get('appointment_details', {})
            
            appointments_list.append({
                "id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "provider": appointment_details.get('doctor', 'TBD'),
                "specialty": appointment_details.get('specialty', 'General'),
                "date": appointment_details.get('date', 'TBD'),
                "time": appointment_details.get('time', 'TBD'),
                "type": "EmailJS AI Consultation",
                "location": appointment_details.get('location', 'TBD'),
                "status": appt.get('status', 'pending'),
                "urgency": appt.get('urgency', 'routine'),
                "emailjs_ready": appt.get('emailjs_ready', False),
                "emailjs_data": appt.get('emailjs_data', {}),
                "processing_summary": appt.get('processing_summary', ''),
                "created_at": appt.get('created_at', '')
            })
        
        return {
            "appointments": appointments_list,
            "total": len(appointments_list),
            "emailjs_ready_count": sum(1 for a in appointments_list if a['emailjs_ready']),
            "emailjs_success_rate": sum(1 for a in appointments_list if a['emailjs_ready']) / max(len(appointments_list), 1) * 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emailjs-logs")
async def get_emailjs_logs():
    """Get EmailJS preparation and delivery logs"""
    try:
        logs_list = []
        for patient_id, log in emailjs_logs.items():
            patient = patients_db.get(patient_id, {})
            logs_list.append({
                "patient_id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "patient_email": log['patient_email'],
                "emailjs_data_prepared": log['emailjs_data_prepared'],
                "emailjs_data": log.get('emailjs_data', {}),
                "timestamp": log['timestamp']
            })
        
        return {
            "emailjs_logs": logs_list,
            "total_preparations": len(logs_list),
            "successful_preparations": sum(1 for log in logs_list if log['emailjs_data_prepared']),
            "preparation_rate": sum(1 for log in logs_list if log['emailjs_data_prepared']) / max(len(logs_list), 1) * 100,
            "email_service": "EmailJS"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-slots")
async def get_emailjs_available_slots(date: str, provider: Optional[str] = None):
    """Get available time slots with EmailJS compatibility"""
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
            "emailjs_compatible": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{patient_id}")
async def get_emailjs_medical_report(patient_id: str):
    """Download EmailJS-optimized medical report PDF"""
    try:
        if patient_id not in reports_db:
            raise HTTPException(status_code=404, detail="EmailJS medical report not found")
        
        report_info = reports_db[patient_id]
        pdf_path = report_info["report_path"]
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=pdf_path,
            filename=f"emailjs_medical_report_{patient_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/health")
async def emailjs_health_check():
    """EmailJS system health check"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "email_service": "EmailJS",
        "timestamp": datetime.now().isoformat(),
        "patients_count": len(patients_db),
        "appointments_count": len(appointments_db),
        "reports_count": len(reports_db),
        "emailjs_preparations": len(emailjs_logs),
        "emailjs_success_rate": sum(1 for log in emailjs_logs.values() if log['emailjs_data_prepared']) / max(len(emailjs_logs), 1) * 100,
        "features": [
            "EmailJS Integration",
            "Enhanced CrewAI Integration",
            "Automatic Email Preparation", 
            "Date/Time Preferences",
            "Comprehensive PDF Reports",
            "All AI Agents Utilized"
        ]
    }

@app.post("/system/reset")
async def reset_emailjs_system():
    """Reset all EmailJS system data"""
    global appointments_db, patients_db, reports_db, emailjs_logs
    appointments_db.clear()
    patients_db.clear()
    reports_db.clear()
    emailjs_logs.clear()
    
    return {
        "success": True,
        "message": "EmailJS system data reset successfully",
        "email_service": "EmailJS",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting EmailJS Healthcare AI FastAPI Server...")
    print("✨ EmailJS Features:")
    print("   - Frontend EmailJS Integration")
    print("   - No SMTP Configuration Required")
    print("   - Browser-based Email Delivery")
    print("   - Enhanced PDF Reports")
    print("   - Comprehensive Medical Analysis")
    print("\n📋 EmailJS Endpoints:")
    print("   - POST /process-patient-enhanced")
    print("   - POST /medical-analysis-enhanced") 
    print("   - GET  /emailjs-logs")
    print("   - GET  /providers (EmailJS ready)")
    print("   - GET  /appointments (EmailJS enhanced)")
    print("\n🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n📧 EmailJS Setup Required:")
    print("   - Create EmailJS account")
    print("   - Set up email service")
    print("   - Configure environment variables")
    
    uvicorn.run(
        "emailjs_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
