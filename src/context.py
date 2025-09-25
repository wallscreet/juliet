from typing import List
import chromadb
from chromadb.utils import embedding_functions
import yaml
from messages import Turn


def format_chat_history(chat_history: list):
    """
    Formats the chat history for display.
    """
    formatted_history = ''.join(chat_history)
    formatted_history = formatted_history.replace('\n\n', '\n')

    return formatted_history


class MemoryAdapter:
    """
    Abstract interface for long-term memory backends. 
    *Note: AI suggested on review, unsure of exact use yet, haha
    """

    def store_turn(self, conversation_id: str, turn: Turn):
        raise NotImplementedError

    def retrieve_turns(self, query: str, top_k: int = 5) -> List[Turn]:
        raise NotImplementedError


class ChromaMemoryAdapter(MemoryAdapter):
    """
    ChromaDB-based long-term memory.
    Supports multiple collections like facts, history, summaries, etc..
    """

    def __init__(self, persist_dir: str = "./chroma_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collections = {}

    def _get_collection(self, name: str):
        """
        Create or fetch a named collection (e.g., 'facts', 'history', 'summaries').
        """
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_fn
            )
        return self.collections[name]

    def store_turn(self, conversation_id: str, turn: Turn, collection_name: str = "history"):
        """
        Store a turn in the specified collection.
        """
        collection = self._get_collection(collection_name)

        # TODO: should I add request and response as separate docs or combined. starting with separate ??
        docs = [
            {
                "id": f"{turn.uuid}_req",
                "text": turn.request.to_memory_string(),
                "metadata": {
                    "conversation_id": conversation_id,
                    "role": turn.request.role,
                    "speaker": turn.request.speaker,
                    "timestamp": turn.request.timestamp,
                    "tags": turn.request.tags,
                },
            },
            {
                "id": f"{turn.uuid}_res",
                "text": turn.response.to_memory_string(),
                "metadata": {
                    "conversation_id": conversation_id,
                    "role": turn.response.role,
                    "speaker": turn.response.speaker,
                    "timestamp": turn.response.timestamp,
                    "tags": turn.response.tags,
                },
            }
        ]

        collection.add(
            documents=[doc["text"] for doc in docs],
            ids=[doc["id"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs]
        )

    def retrieve(self, query: str, top_k: int = 5, collection_name: str = "history") -> List[dict]:
        """
        Retrieve semantically relevant docs from a collection.
        Returns metadata + content.
        """
        collection = self._get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)

        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({"text": doc, "metadata": meta})
        return retrieved


class YamlMemoryAdapter(MemoryAdapter):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def store_turn(self, conversation_id: str, turn: Turn):
        turn_dict = turn.to_dict()

        with open(self.filepath, 'r') as file:
            data = yaml.safe_load(file) or {"conversations": []}

        conversation = next((c for c in data["conversations"] if c["uuid"] == conversation_id), None)
        if conversation:
            conversation["turns"].append(turn_dict)
            conversation["last_active"] = turn_dict["response"]["timestamp"]
        else:
            conversation = {
                "uuid": conversation_id,
                "created_at": turn_dict["request"]["timestamp"],
                "last_active": turn_dict["response"]["timestamp"],
                "turns": [turn_dict]
            }
            data["conversations"].append(conversation)

        with open(self.filepath, 'w') as file:
            yaml.safe_dump(data, file)
        
    def load_conversation_by_id(self, conversation_id):
        pass

    def retrieve(self, query: str, top_k: int = 10) -> List[Turn]:
        """
        Naive retrieval: scan YAML conversations and return last K turns.
        """
        with open(self.filepath, 'r') as file:
            data = yaml.safe_load(file) or {"conversations": []}

        turns = []
        for convo in data["conversations"]:
            for turn in convo.get("turns", []):
                turns.append(turn)

        return turns[-top_k:]

