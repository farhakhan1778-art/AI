# vector_store.py
# Phase 3 — ChromaDB Vector Database with sentence-transformers all-MiniLM-L6-v2
# Stores embeddings, metadata, description, eligibility, benefits, and links for semantic retrieval.

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config

class SchemeVectorStore:
    def __init__(self):
        config.CHROMA_DIR.mkdir(exist_ok=True)
        print(f"Initializing ChromaDB client at: {config.CHROMA_DIR}")
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generates 384-dim dense vector embedding using sentence-transformers."""
        return self.embedding_model.encode(text).tolist()

    def build_database(self, force_rebuild: bool = True):
        """Loads data/schemes.json, generates embeddings, and populates ChromaDB."""
        if not config.RAW_SCHEMES_PATH.exists():
            raise FileNotFoundError(f"Missing schemes data at {config.RAW_SCHEMES_PATH}. Run Phase 1 & Phase 2 first.")

        with open(config.RAW_SCHEMES_PATH, "r", encoding="utf-8") as f:
            schemes = json.load(f)

        if force_rebuild:
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        print(f"Building ChromaDB index for {len(schemes)} government schemes...")

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for idx, scheme in enumerate(schemes):
            scheme_id = scheme.get("scheme_id", f"SCH_{idx:03d}")
            name = scheme.get("scheme_name", "")
            desc = scheme.get("description", "")
            state = scheme.get("state", "Central")
            ministry = scheme.get("ministry", "")
            website = scheme.get("official_website", "")
            app_process = scheme.get("application_process", "")
            benefits_list = scheme.get("benefits", [])
            benefits_str = " ".join(benefits_list) if isinstance(benefits_list, list) else str(benefits_list)
            
            elig = scheme.get("eligibility", {})
            meta = scheme.get("metadata", {})
            
            # Combine text representations for dense embedding
            doc_text = f"""
Scheme Name: {name}
State: {state}
Ministry: {ministry}
Description: {desc}
Eligibility: Age {elig.get('age_min', 0)}-{elig.get('age_max', 100)}, Gender: {', '.join(elig.get('genders', ['All']))}, Category: {', '.join(elig.get('categories', ['All']))}, Occupation: {', '.join(elig.get('occupations', ['All']))}, Income: {', '.join(elig.get('income_brackets', ['Any']))}
Benefits: {benefits_str}
Required Documents: {', '.join(scheme.get('required_documents', []))}
Application Process: {app_process}
Official Website: {website}
""".strip()

            # Format ChromaDB metadata (strings/numbers only)
            occs = meta.get("occupation", elig.get("occupations", ["All"]))
            cats = meta.get("category", elig.get("categories", ["All"]))
            
            occ_str = ",".join([o.lower() for o in occs]) if isinstance(occs, list) else str(occs).lower()
            cat_str = ",".join([c.lower() for c in cats]) if isinstance(cats, list) else str(cats).lower()
            
            chroma_meta = {
                "scheme_id": scheme_id,
                "scheme_name": name,
                "state": state.lower(),
                "gender": str(meta.get("gender", "all")).lower(),
                "income": str(meta.get("income", "below1l")).lower(),
                "benefit_type": str(meta.get("benefit_type", "financial")).lower(),
                "occupations": occ_str,
                "categories": cat_str,
                "official_website": website,
                "age_min": int(elig.get("age_min", 0)),
                "age_max": int(elig.get("age_max", 100)),
                "raw_json": json.dumps(scheme, ensure_ascii=False)
            }

            emb = self.generate_embedding(doc_text)

            documents.append(doc_text)
            metadatas.append(chroma_meta)
            ids.append(scheme_id)
            embeddings.append(emb)

        # Batch insert into ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

        print(f"[OK] ChromaDB store updated successfully. Total documents indexed: {self.collection.count()}")

    def query_schemes(self, query: str, user_profile: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB vector database using semantic search and user profile metadata filters.
        """
        query_emb = self.generate_embedding(query)
        
        # Build ChromaDB metadata filter `where` clause
        where_clause = {}
        conditions = []

        if user_profile:
            state = user_profile.get("state")
            if state and state.lower() not in ["all", "other", "none"]:
                # State filter: Match user state OR Central schemes
                conditions.append({
                    "$or": [
                        {"state": {"$eq": state.lower()}},
                        {"state": {"$eq": "central"}}
                    ]
                })

            gender = user_profile.get("gender")
            if gender and gender.lower() in ["male", "female", "transgender"]:
                conditions.append({
                    "$or": [
                        {"gender": {"$eq": gender.lower()}},
                        {"gender": {"$eq": "all"}}
                    ]
                })

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # Perform query on ChromaDB
        query_kwargs = {
            "query_embeddings": [query_emb],
            "n_results": top_k
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as e:
            print(f"ChromaDB filtered query warning: {e}. Retrying without metadata filters.")
            results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)

        candidates = []
        if results and "metadatas" in results and results["metadatas"]:
            for i, meta in enumerate(results["metadatas"][0]):
                raw_data = json.loads(meta.get("raw_json", "{}"))
                candidates.append(raw_data)

        return candidates

if __name__ == "__main__":
    print("=== PHASE 3: VECTOR DATABASE INITIALIZATION ===")
    store = SchemeVectorStore()
    store.build_database(force_rebuild=True)
    res = store.query_schemes("farmer crop subsidy", user_profile={"state": "Rajasthan", "gender": "Male"}, top_k=3)
    print(f"Test query returned {len(res)} results.")
    if res:
        print(f"Top result: {res[0].get('scheme_name')} ({res[0].get('state')})")
    print("=== PHASE 3 COMPLETED SUCCESSFULLY ===")
