from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from collections import deque
import yaml


@dataclass
class Message:
    uuid: str # The uuid may actually be arbitrary here
    role: str
    speaker: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d @ %H:%M'))

    tags: List[str] = field(default_factory=list)  
    embedding: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)

    def to_prompt_message_string(self) -> str:
        return f"<|im_start|>{self.speaker}:\n{self.content}<|im_end|>"

    def to_memory_string(self) -> str:
        return f"{self.speaker} @ {self.timestamp}: {self.content}"


@dataclass
class Turn:
    uuid: str
    conversation_id: str
    request: Message
    response: Message

    def to_dict(self):
        return asdict(self)

    def to_memory_string(self) -> dict:
        """
        Convert a turn to a prompt friendly format.
        """
        return {
            "request": self.request.to_memory_string(),
            "response": self.response.to_memory_string(),
        }


@dataclass
class Conversation:
    uuid: str
    description: str
    created_at: str
    last_active: str
    host: str
    host_is_bot: bool
    guest: str
    guest_is_bot: bool
    turns: List[Turn] = field(default_factory=list)

    def create_turn(self, conversation_id: str, request: Message, response: Message) -> Turn:
        turn = Turn(uuid=str(uuid4()), request=request, response=response, conversation_id=conversation_id)
        self.turns.append(turn)
        self.last_active = response.timestamp
        
        return turn

    def to_dict(self):
        return {
            **asdict(self),
            "turns": [
                {
                    **asdict(turn),
                    "request": asdict(turn.request),
                    "response": asdict(turn.response)
                } for turn in self.turns
            ]
        }

    def save_to_yaml(self, yaml_path: str):
        try:
            # Load existing conversations if file exists
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or []
        except FileNotFoundError:
            data = []

        # overwrite entry with same uuid??
        data = [c for c in data if c['uuid'] != self.uuid]
        data.append(self.to_dict())

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data, f)
        print(f"Conversation {self.uuid} saved to {yaml_path}.")

    @classmethod
    def start_new(cls, host: str, host_is_bot: bool, guest: str, guest_is_bot: bool, uuid_override: Optional[str] = None) -> "Conversation":
        now = datetime.now().strftime('%Y-%m-%d @ %H:%M')
        conversation = cls(
            uuid=uuid_override or str(uuid4()),
            description=f"{host}-{guest}",
            created_at=now,
            last_active=now,
            host=host,
            host_is_bot=host_is_bot,
            guest=guest,
            guest_is_bot=guest_is_bot,
            turns=[]
        )
        print(f"New conversation {conversation.description} | {conversation.uuid} started!")
        return conversation

    @classmethod
    def load_from_yaml(cls, yaml_path: str, conversation_id: str) -> "Conversation":
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or []

        conv_data = next((c for c in data if c['uuid'] == conversation_id), None)
        if not conv_data:
            raise ValueError(f"No conversation with id {conversation_id} found")

        conv_data['turns'] = [
            Turn(
                uuid=t['uuid'],
                request=Message(**t['request']),
                response=Message(**t['response'])
            ) for t in conv_data.get('turns', [])
        ]

        return cls(**conv_data)


# TODO: this message cache get chat history will output a different format than expected. should probably refactor this to return the turns and messages or the other side to take the list of strings.
class MessageCache:
    """
    A bounded short-term memory buffer (e.g. last N turns).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = deque(maxlen=capacity)

    def add_turn(self, turn: Turn):
        self.cache.append(turn)
    
    def get_message_cache(self):
        message_cache = list(self.cache)
        return message_cache

    def get_n_turns(self, n: int) -> List[Turn]:
        return list(self.cache)[-n:]

    def get_chat_history(self, as_strings=True):
        if as_strings:
            history = []
            for turn in self.cache:
                history.append(turn.request.to_memory_string())
                history.append(turn.response.to_memory_string())
            return history
        return list(self.cache)
