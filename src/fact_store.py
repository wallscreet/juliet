from datetime import datetime
from uuid import uuid4
import chromadb
from pydantic import BaseModel
import yaml
from typing import List
import os

from context import ChromaMemoryAdapter
from messages import Message, Turn


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str

    def to_memory_string(self) -> str:
        """
        Serialize the fact to a string in the format 'subject predicate object'.
        """
        return f"{self.subject} {self.predicate} {self.object}"


class FactStore:
    def __init__(self, file_path: str = "facts.yaml", chroma_adapter: ChromaMemoryAdapter = None):
        self.file_path = file_path
        self.chroma_adapter = chroma_adapter or ChromaMemoryAdapter(persist_dir="./chroma_store")
        
        # Initialize YAML file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                yaml.dump({'facts': []}, f)

    def append_fact(self, fact: Fact):
        # Append to YAML file
        with open(self.file_path, 'r') as f:
            data = yaml.safe_load(f) or {'facts': []}
        
        data['facts'].append(fact.model_dump())
        
        with open(self.file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        self.store_fact(fact=fact)
    
    def store_fact(self, fact: Fact, collection_name: str = "facts"):
        """
        Store a single fact as a subject-predicate-object triple in the specified collection.
        """
        collection = self.chroma_adapter._get_collection(name=collection_name)

        fact_id = str(uuid4())  # Unique ID for the fact
        fact_str = f"{fact.subject} {fact.predicate} {fact.object}"  # Serialize fact for embedding
        
        collection.add(
            documents=[fact_str],
            ids=[fact_id],
            metadatas=[{
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "timestamp": datetime.now().strftime('%Y-%m-%d @ %H:%M'),
                "type": "fact"
            }]
        )
    
    def retrieve_facts(self, query: str, collection_name: str = "facts", top_k: int = 10) -> List[Fact]:
        """
        Retrieve semantically relevant facts from the specified collection.
        Returns a list of Fact objects constructed from metadata.
        """
        collection = self.chroma_adapter._get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)

        facts = []
        for meta in results["metadatas"][0]:
            if meta.get("type") == "fact":
                facts.append(
                    Fact(
                        subject=meta.get("subject", ""),
                        predicate=meta.get("predicate", ""),
                        object=meta.get("object", "")
                    )
                )
        return facts
