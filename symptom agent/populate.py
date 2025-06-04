from neo4j import GraphDatabase
import pandas as pd
import os

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_knowledge_graph(csv_path):
    df = pd.read_csv(csv_path)
    symptoms = list(df.columns)[1:]

    with driver.session() as session:
        # Clean the DB first (optional)
        print("[1/3] Deleting all previous nodes and relationships from the database...")
        session.run("MATCH (n) DETACH DELETE n")
        print("   --> Database cleaned.")

        print("[2/3] Populating graph from CSV...")
        for idx, row in df.iterrows():
            disease = row['diseases'].strip()
            # Create Disease node
            session.run("MERGE (d:Disease {name: $name})", name=disease)
            created_symptoms = 0
            created_relationships = 0
            for i, val in enumerate(row[1:]):
                if val == 1:
                    symptom = symptoms[i]
                    # Create Symptom node
                    session.run("MERGE (s:Symptom {name: $symptom})", symptom=symptom)
                    # Create relationship
                    session.run("""
                        MATCH (d:Disease {name: $disease}), (s:Symptom {name: $symptom})
                        MERGE (d)-[:HAS_SYMPTOM]->(s)
                    """, disease=disease, symptom=symptom)
                    created_symptoms += 1
                    created_relationships += 1
            print(f"   [{idx+1}/{len(df)}] Disease '{disease}': {created_symptoms} symptoms, {created_relationships} relationships created.")
        print("[3/3] Graph build complete!")

#create_knowledge_graph(r"D:\Agentic Ai\Health-Planner\symptom checker\reduced_disease_dataset.csv")

