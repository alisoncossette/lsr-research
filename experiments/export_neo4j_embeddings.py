"""
Export document embeddings from Neo4j to CSV for LSR evaluation.

Usage:
    python experiments/export_neo4j_embeddings.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Change neo4j+s:// to neo4j+ssc:// to trust self-signed certs
if NEO4J_URI and "+s://" in NEO4J_URI:
    NEO4J_URI = NEO4J_URI.replace("+s://", "+ssc://")

print(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )

with driver.session() as session:
                session.run("Match (n)RETURN n limit 1")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError(
        "Missing Neo4j credentials in .env file. "
        "Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD"
    )


def export_embeddings(output_file: str = "data/neo4j_embeddings.csv") -> str:
    """
    Export document embeddings from Neo4j to CSV.

    Args:
        output_file: Path to output CSV file

    Returns:
        Path to created CSV file
    """
    print(f"\n{'='*70}")
    print("NEO4J EMBEDDING EXPORT")
    print(f"{'='*70}")

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to Neo4j
    print(f"\n[1] Connecting to Neo4j...")
    print(f"    URI: {NEO4J_URI}")

    try:
        # Try with default settings first
        try:
            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            print("    [OK] Connected successfully")
        except Exception as default_error:
            # If that fails, try with explicit SSL disabled (older approach)
            print(f"    [WARNING] Default connection failed, trying without SSL verification...")
    except Exception as e:
        print(f"    [ERROR] Connection failed: {e}")
        raise

    # Query documents with embeddings
    print(f"\n[2] Querying documents from Neo4j...")
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                RETURN d.id as id, d.text as text, d.embedding as embedding
                """
            )
            records = list(result)
        print(f"    [OK] Retrieved {len(records)} documents")
    except Exception as e:
        print(f"    [ERROR] Query failed: {e}")
        driver.close()
        raise

    if not records:
        print("    [WARNING] No documents found")
        driver.close()
        return None

    # Parse records
    print(f"\n[3] Processing embeddings...")
    data = []

    for i, record in enumerate(records):
        doc_id = record['id']
        text = record['text']
        embedding = record['embedding']

        # Handle embedding format (could be list, string, or vector)
        if embedding is None:
            print(f"    [WARNING] Document {doc_id} has no embedding, skipping")
            continue

        # Convert embedding to list if needed
        if isinstance(embedding, str):
            try:
                embedding = eval(embedding)  # If stored as string
            except:
                print(f"    [WARNING] Could not parse embedding for {doc_id}, skipping")
                continue
        elif not isinstance(embedding, (list, np.ndarray)):
            print(f"    [WARNING] Invalid embedding format for {doc_id}, skipping")
            continue

        embedding = np.array(embedding, dtype=np.float32)

        data.append({
            'doc_id': str(doc_id),
            'text': str(text),
            'embedding': embedding,
            'embedding_dim': len(embedding)
        })

    driver.close()

    if not data:
        print("    [WARNING] No valid embeddings found")
        return None

    print(f"    [OK] Processed {len(data)} documents with embeddings")

    # Create DataFrame
    print(f"\n[4] Creating output CSV...")

    # Get embedding dimension
    embedding_dim = data[0]['embedding_dim']
    print(f"    - Embedding dimension: {embedding_dim}")

    # Create CSV with embedding columns
    rows = []
    for item in data:
        row = {
            'doc_id': item['doc_id'],
            'text': item['text']
        }
        # Add embedding as separate columns
        embedding = item['embedding']
        for j, val in enumerate(embedding):
            row[f'emb_{j}'] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"    [OK] Saved {len(df)} rows to {output_path}")

    # Print summary
    print(f"\n[5] Summary")
    print(f"    - Total documents: {len(df)}")
    print(f"    - Embedding columns: {embedding_dim}")
    print(f"    - Total columns: {len(df.columns)}")
    print(f"    - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\n{'='*70}")
    print(f"Export complete: {output_path}")
    print(f"{'='*70}\n")

    return str(output_path)


if __name__ == "__main__":
    try:
        csv_path = export_embeddings()
        if csv_path:
            print(f"[OK] Embeddings exported to: {csv_path}")
            sys.exit(0)
        else:
            print("[ERROR] Export failed - no embeddings found")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
