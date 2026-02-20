"""
Neo4j Integration for LSR Experiments

Provides utilities to:
- Connect to Neo4j database
- Retrieve documents and queries
- Manage embeddings (generate, cache, retrieve)
- Run LSR experiments on real data
- Visualize results with t-SNE
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm

try:
    from neo4j import GraphDatabase, Session
except ImportError:
    raise ImportError("neo4j package required. Install with: pip install neo4j")

from .lsr import LocalSpectralRetrieval, TopKRetrieval, ThresholdRandomRetrieval, MMRRetrieval
from .metrics import evaluate_retrieval
from .embeddings import ClaudeEmbedder


@dataclass
class DocumentRecord:
    """Represents a document from Neo4j"""
    doc_id: str
    text: str
    metadata: Dict = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            'id': self.doc_id,
            'text': self.text,
            'metadata': self.metadata or {}
        }


@dataclass
class QueryRecord:
    """Represents a query with relevant documents"""
    query_id: str
    query_text: str
    embedding: Optional[np.ndarray] = None
    relevant_doc_ids: Set[str] = None

    def __post_init__(self):
        if self.relevant_doc_ids is None:
            self.relevant_doc_ids = set()


class Neo4jConnector:
    """Connect to and interact with Neo4j database"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection.

        Args:
            uri: Connection URI (e.g., "bolt://localhost:7687")
            username: Database username
            password: Database password
            database: Database name (default: "neo4j")
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                database=database
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✓ Connected to Neo4j at {uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

    def run_query(self, cypher: str, parameters: Dict = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def get_documents(self, node_label: str, text_property: str,
                     id_property: str = "id", limit: Optional[int] = None) -> List[DocumentRecord]:
        """
        Retrieve documents from Neo4j.

        Args:
            node_label: Node label (e.g., "Document")
            text_property: Property containing document text
            id_property: Property containing document ID (default: "id")
            limit: Maximum number of documents to retrieve (None = all)

        Returns:
            List of DocumentRecord objects
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        cypher = f"""
        MATCH (d:{node_label})
        RETURN d.{id_property} as id, d.{text_property} as text, properties(d) as metadata
        {limit_clause}
        """

        results = self.run_query(cypher)
        documents = []

        for record in results:
            doc = DocumentRecord(
                doc_id=str(record['id']),
                text=str(record['text']),
                metadata=record.get('metadata', {})
            )
            documents.append(doc)

        return documents

    def get_document_by_id(self, node_label: str, doc_id: str,
                         text_property: str, id_property: str = "id") -> Optional[DocumentRecord]:
        """
        Retrieve a single document by ID.

        Args:
            node_label: Node label (e.g., "Document")
            doc_id: Document ID
            text_property: Property containing document text
            id_property: Property containing document ID

        Returns:
            DocumentRecord or None if not found
        """
        cypher = f"""
        MATCH (d:{node_label})
        WHERE d.{id_property} = $doc_id
        RETURN d.{id_property} as id, d.{text_property} as text, properties(d) as metadata
        LIMIT 1
        """

        results = self.run_query(cypher, {'doc_id': doc_id})
        if not results:
            return None

        record = results[0]
        return DocumentRecord(
            doc_id=str(record['id']),
            text=str(record['text']),
            metadata=record.get('metadata', {})
        )

    def count_documents(self, node_label: str) -> int:
        """Count total documents of a given type"""
        cypher = f"MATCH (d:{node_label}) RETURN count(d) as count"
        results = self.run_query(cypher)
        return results[0]['count'] if results else 0


class Neo4jEmbeddingCache:
    """Cache embeddings locally to avoid regenerating them"""

    def __init__(self, cache_file: Path = Path("data/embeddings_cache.json")):
        """
        Initialize embedding cache.

        Args:
            cache_file: Path to JSON file for caching embeddings
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            cache_to_save = {}
            for key, value in self.cache.items():
                if isinstance(value, np.ndarray):
                    cache_to_save[key] = value.tolist()
                else:
                    cache_to_save[key] = value
            json.dump(cache_to_save, f)

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if key in self.cache:
            value = self.cache[key]
            if isinstance(value, list):
                return np.array(value)
            return value
        return None

    def set(self, key: str, embedding: np.ndarray):
        """Store embedding in cache"""
        self.cache[key] = embedding
        self._save_cache()

    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self._save_cache()


class Neo4jLSRExperiment:
    """Run LSR experiments using documents from Neo4j"""

    def __init__(self, neo4j_uri: str, username: str, password: str,
                 node_label: str = "Document", text_property: str = "text",
                 id_property: str = "id", database: str = "neo4j"):
        """
        Initialize LSR experiment with Neo4j backend.

        Args:
            neo4j_uri: Neo4j connection URI
            username: Database username
            password: Database password
            node_label: Node label for documents
            text_property: Property containing text
            id_property: Property containing document ID
            database: Database name
        """
        self.connector = Neo4jConnector(neo4j_uri, username, password, database)
        self.node_label = node_label
        self.text_property = text_property
        self.id_property = id_property

        self.documents: List[DocumentRecord] = []
        self.queries: List[QueryRecord] = []
        self.embedder = ClaudeEmbedder()
        self.embedding_cache = Neo4jEmbeddingCache()

        self.methods = {
            'LSR (quantile)': LocalSpectralRetrieval(sampling_method='quantile'),
            'LSR (deterministic)': LocalSpectralRetrieval(sampling_method='deterministic'),
            'Top-k': TopKRetrieval(),
            'Threshold+Random': ThresholdRandomRetrieval(),
            'MMR': MMRRetrieval()
        }

        self.results = []

    def load_documents(self, limit: Optional[int] = None) -> int:
        """
        Load documents from Neo4j.

        Args:
            limit: Maximum documents to load (None = all)

        Returns:
            Number of documents loaded
        """
        print(f"\nLoading documents from Neo4j...")
        self.documents = self.connector.get_documents(
            self.node_label,
            self.text_property,
            self.id_property,
            limit=limit
        )
        print(f"✓ Loaded {len(self.documents)} documents")
        return len(self.documents)

    def generate_embeddings(self, batch_size: int = 10, skip_cached: bool = True):
        """
        Generate embeddings for documents.

        Args:
            batch_size: Documents to process per batch
            skip_cached: Skip documents with cached embeddings
        """
        print(f"\nGenerating embeddings for documents...")

        docs_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(self.documents):
            if skip_cached and self.embedding_cache.has(doc.doc_id):
                # Load from cache
                doc.embedding = self.embedding_cache.get(doc.doc_id)
            else:
                docs_to_embed.append(doc.text)
                indices_to_embed.append(i)

        if not docs_to_embed:
            print("✓ All documents already cached")
            return

        # Generate embeddings in batches
        print(f"  - Generating {len(docs_to_embed)} new embeddings...")

        for i in tqdm(range(0, len(docs_to_embed), batch_size)):
            batch = docs_to_embed[i:i+batch_size]
            batch_indices = indices_to_embed[i:i+batch_size]

            # Generate embeddings for batch
            embeddings = self.embedder.embed_texts(batch)

            # Cache and store
            for emb, idx in zip(embeddings, batch_indices):
                self.documents[idx].embedding = emb
                self.embedding_cache.set(self.documents[idx].doc_id, emb)

        print("✓ Embeddings complete")

    def add_query(self, query_id: str, query_text: str,
                 relevant_doc_ids: Optional[Set[str]] = None):
        """
        Add a query to the experiment.

        Args:
            query_id: Unique query identifier
            query_text: Query text
            relevant_doc_ids: Set of relevant document IDs
        """
        query = QueryRecord(
            query_id=query_id,
            query_text=query_text,
            relevant_doc_ids=relevant_doc_ids or set()
        )
        self.queries.append(query)

    def run_experiment(self, k: int = 10, threshold: float = 0.3):
        """
        Run LSR experiment on loaded queries.

        Args:
            k: Number of documents to retrieve
            threshold: Similarity threshold for initial retrieval
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first")
        if not self.queries:
            raise ValueError("No queries added. Call add_query() first")

        # Check embeddings
        if any(doc.embedding is None for doc in self.documents):
            print("Generating missing embeddings...")
            self.generate_embeddings()

        for query in self.queries:
            if query.embedding is None:
                query.embedding = self.embedder.embed_text(query.query_text)

            # Build corpus
            corpus = np.array([doc.embedding for doc in self.documents])

            # Convert relevant doc IDs to indices
            doc_id_to_idx = {doc.doc_id: i for i, doc in enumerate(self.documents)}
            relevant_indices = set(
                doc_id_to_idx[doc_id]
                for doc_id in query.relevant_doc_ids
                if doc_id in doc_id_to_idx
            )

            # Run each retrieval method
            for method_name, retriever in self.methods.items():
                try:
                    # Retrieve
                    embeddings, indices = retriever.retrieve(
                        query.embedding,
                        corpus,
                        k=k,
                        threshold=threshold
                    )

                    # Evaluate
                    metrics = evaluate_retrieval(
                        embeddings,
                        indices,
                        relevant_indices,
                        k=k
                    )

                    # Store result
                    self.results.append({
                        'query_id': query.query_id,
                        'method': method_name,
                        'k': k,
                        'threshold': threshold,
                        'metrics': metrics
                    })

                except Exception as e:
                    print(f"  ! {method_name} failed: {e}")

    def get_results(self) -> List[Dict]:
        """Get all experiment results"""
        return self.results

    def save_results(self, output_file: Path = Path("results/neo4j_results.json")):
        """Save results to JSON"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists
        results_to_save = []
        for result in self.results:
            result_copy = result.copy()
            # metrics might have numpy values
            result_copy['metrics'] = {k: float(v) if isinstance(v, (np.ndarray, np.number)) else v
                                     for k, v in result['metrics'].items()}
            results_to_save.append(result_copy)

        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"✓ Results saved to {output_file}")

    def close(self):
        """Close database connection"""
        self.connector.close()


def create_experiment_from_config(config_file: Path) -> Neo4jLSRExperiment:
    """
    Create experiment from configuration file.

    Expected config.json structure:
    {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
        },
        "documents": {
            "node_label": "Document",
            "text_property": "text",
            "id_property": "id"
        }
    }
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    neo4j_config = config['neo4j']
    doc_config = config.get('documents', {})

    return Neo4jLSRExperiment(
        neo4j_uri=neo4j_config['uri'],
        username=neo4j_config['username'],
        password=neo4j_config['password'],
        node_label=doc_config.get('node_label', 'Document'),
        text_property=doc_config.get('text_property', 'text'),
        id_property=doc_config.get('id_property', 'id'),
        database=neo4j_config.get('database', 'neo4j')
    )
