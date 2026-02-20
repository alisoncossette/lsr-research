// Neo4j Cypher query to export document embeddings
// Copy and paste this into Neo4j Browser and execute

// Get all documents with their embeddings
MATCH (d:Document)
RETURN d.id as doc_id, d.text as text, d.embedding as embedding
LIMIT 10000;

// Alternative: If you want to export as CSV from Neo4j Browser:
// 1. Run the above query
// 2. Click the "Download" button in the results panel
// 3. Select CSV format

// Alternative query if you need metadata too:
MATCH (d:Document)
RETURN
  d.id as doc_id,
  d.text as text,
  d.embedding as embedding,
  d.created_date as created_date
LIMIT 10000;
