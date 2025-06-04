from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Create driver instance
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_diseases_from_neo4j(user_symptoms, top_n=5):
    """
    Query Neo4j to find diseases matching the given symptoms.
    
    Args:
        user_symptoms (list): List of symptom strings
        top_n (int): Maximum number of diseases to return
    
    Returns:
        list: List of dictionaries with disease information
    """
    with driver.session() as session:
        try:
            result = session.run("""
                MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
                WHERE toLower(s.name) IN $symptom_list
                WITH d, collect(s.name) as matched_symptoms, count(*) as match_count
                ORDER BY match_count DESC
                RETURN d.name as disease, matched_symptoms, match_count
                LIMIT $top_n
            """, symptom_list=[s.lower().strip() for s in user_symptoms], top_n=top_n)
            
            return [
                {
                    "disease": record["disease"],
                    "matched_symptoms": record["matched_symptoms"],
                    "match_count": record["match_count"]
                }
                for record in result
            ]
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            return []

def get_all_symptoms_from_neo4j():
    """
    Retrieve all available symptoms from Neo4j database.
    
    Returns:
        list: List of all symptom names
    """
    with driver.session() as session:
        try:
            result = session.run("MATCH (s:Symptom) RETURN s.name as symptom ORDER BY s.name")
            return [record["symptom"] for record in result]
        except Exception as e:
            print(f"Error retrieving symptoms: {e}")
            return []

def close_driver():
    """Close the Neo4j driver connection."""
    if driver:
        driver.close()



