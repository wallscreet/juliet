from datetime import datetime
from typing import List, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
import yaml
from messages import Conversation, Message, Turn


def format_chat_history(chat_history: list):
    """
    Formats the chat history for display. This didn't really solve anything..
    """
    formatted_history = ''.join(chat_history)
    formatted_history = formatted_history.replace('\n\n', '\n')

    return formatted_history

# TODO: Do i even need this??? If i return the list of strings from the other side then I can just push that directly into the prompt content string
def message_cache_format_to_prompt(message_history):
    chat_history = []
    for turn in message_history:
        turn_request = f"{turn.request.speaker} ({turn.request.timestamp}):\n{turn.request.content}\n"
        turn_response = f"{turn.response.speaker} ({turn.response.timestamp}):\n{turn.response.content}\n"
        chat_history.append(turn_request)
        chat_history.append(turn_response)
    chat_history = format_chat_history(chat_history)
    #print(f"\n{chat_history}")
    return chat_history


class MemoryAdapter:
    """
    Abstract interface for long-term memory backends. 
    *Note: AI suggested on review, unsure of exact use yet, haha
    """

    def store_turn(self, conversation_id: str, turn: Turn):
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> List[Message]:
        raise NotImplementedError


class ChromaMemoryAdapter(MemoryAdapter):
    """
    ChromaDB-based long-term memory.
    Supports multiple collections like 'history', 'facts', 'summaries', etc.
    """

    def __init__(self, persist_dir: str = "./chroma_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collections = {}

    def _get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_fn
            )
        return self.collections[name]
    
    # TODO: add document knowledge ingestion - pymupdf, text types -> chroma upsert in "knowledge" collection
    def store_knowledge(self):
        pass

    def store_turn(self, conversation_id: str, turn: Turn, collection_name: str = "memory"):
        """
        NOT TESTED
        Store a single turn (request + response as separate docs).
        """
        self.store_batch(conversation_id, [turn], collection_name=collection_name)

    def store_batch(self, conversation_id: str, turns: List[Turn], collection_name: str = "history"):
        """
        NOT TESTED
        Store multiple turns at once (useful for resync / rebuild).
        """
        collection = self._get_collection(collection_name)

        docs, ids, metas = [], [], []
        for turn in turns:
            for suffix, msg in [("req", turn.request), ("res", turn.response)]:
                ids.append(f"{turn.uuid}_{suffix}")
                docs.append(msg.to_memory_string())
                metas.append({
                    "conversation_id": conversation_id,
                    "role": msg.role,
                    "speaker": msg.speaker,
                    "timestamp": msg.timestamp,
                    "tags": msg.tags,
                })

        collection.add(documents=docs, ids=ids, metadatas=metas)

    def retrieve(self, collection_name: str, query: str, top_k: int = 10) -> List[Message]:
        """
        NOT TESTED
        Retrieve semantically relevant messages (normalized to Message schema).
        """
        collection = self._get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)

        messages = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            messages.append(
                Message(
                    uuid=str(uuid4()),
                    role=meta.get("role", "unknown"),
                    speaker=meta.get("speaker", "unknown"),
                    content=text,
                    timestamp=meta.get("timestamp", datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    tags=meta.get("tags", []),
                )
            )
        return messages


class YamlMemoryAdapter(MemoryAdapter):
    """
    Thin wrapper: delegates persistence to Conversation's YAML methods.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def store_turn(self, conversation_id: str, turn: Turn):
        # Load conversation, append turn, save back
        convo = Conversation.load_from_yaml(self.filepath, conversation_id)
        convo.turns.append(turn)
        convo.last_active = turn.response.timestamp
        convo.save_to_yaml(self.filepath)

    def load_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        try:
            return Conversation.load_from_yaml(self.filepath, conversation_id)
        except Exception:
            return None

    def retrieve(self, query: str, top_k: int = 10) -> List[Turn]:
        """
        Naive retrieval: return last N turns across all conversations.
        """
        with open(self.filepath, "r") as f:
            data = yaml.safe_load(f) or []

        turns = []
        for convo in data:
            for t in convo.get("turns", []):
                turns.append(
                    Turn(
                        uuid=t["uuid"],
                        conversation_id=convo["uuid"],
                        turn_number=t.get("turn_number", len(turns) + 1),
                        request=Message(**t["request"]),
                        response=Message(**t["response"]),
                    )
                )
        return turns[-top_k:]
