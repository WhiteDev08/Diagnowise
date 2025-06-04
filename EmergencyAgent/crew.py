from crewai import Crew
from tasks import create_firstaid_task  # Fixed: import the function
from agents import emergency_agent

def main():
    user_input = input("Explain the emergency situation: ")
    
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
    print("FIRST AID GUIDANCE:")
    print("="*50)
    print(results)
    print("="*50)

if __name__ == "__main__":
    main()