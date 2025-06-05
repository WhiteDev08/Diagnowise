from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
import re
from dotenv import load_dotenv

# Import the specific agents and tasks
from EmergencyAgent.agents import emergency_agent
from EmergencyAgent.tasks import create_firstaid_task
from HistoryAgent.agents import medical_history_agent
from HistoryAgent.task import create_history_analysis_task
from SymptomAgent.agents import create_symptom_checker_agent
from SymptomAgent.task import create_diagnosis_task
from SymptomAgent.tools import get_diseases_from_neo4j, get_all_symptoms_from_neo4j, close_driver
from HistoryAgent.pdf_generator import generate_medical_report_pdf

load_dotenv()

class HealthcareRoutingSystem:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.openai_api_key, temperature=0.3)
    
    def display_main_menu(self):
        """Display the main menu options"""
        print("\n" + "="*60)
        print("           🏥 HEALTHCARE ASSISTANCE SYSTEM")
        print("="*60)
        print("Choose your service:")
        print("1. 🚨 Emergency First Aid")
        print("2. 🔍 Symptom Analysis") 
        print("3. 📋 Medical History Analysis")
        print("4. ❌ Exit")
        print("="*60)
    
    def get_user_choice(self):
        """Get and validate user menu choice"""
        while True:
            try:
                choice = input("Enter your choice (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n\nExiting... Stay safe! 👋")
                return 4
            except Exception:
                print("❌ Invalid input. Please enter a number between 1-4.")
    
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
                print("\n👋 Thank you for using Healthcare Assistance System!")
                print("Stay healthy and safe! 💙")
                break
            
            # Ask if user wants to continue
            if choice != 4:
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


