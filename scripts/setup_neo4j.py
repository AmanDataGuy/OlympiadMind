from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

constraints_and_indexes = [
    "CREATE CONSTRAINT problem_id IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.canonical_name IS UNIQUE",
    "CREATE INDEX problem_topic IF NOT EXISTS FOR (p:Problem) ON (p.topic)",
    "CREATE INDEX problem_year IF NOT EXISTS FOR (p:Problem) ON (p.year)",
    "CREATE INDEX problem_competition IF NOT EXISTS FOR (p:Problem) ON (p.competition)",
    "CREATE INDEX problem_source IF NOT EXISTS FOR (p:Problem) ON (p.source)",
    "CREATE INDEX problem_doc_type IF NOT EXISTS FOR (p:Problem) ON (p.doc_type)",
]

with driver.session() as session:
    for query in constraints_and_indexes:
        session.run(query)
        print(f"  OK: {query[:60]}...")

print("\nSchema setup complete.")
driver.close()