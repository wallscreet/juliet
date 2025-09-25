from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from collections import deque
import yaml


@dataclass
class Message:
    uuid: str
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
    turn_number: int
    request: Message
    response: Message

    def to_dict(self):
        return asdict(self)

    def to_context_string(self) -> dict:
        """
        Convert a turn to a context and prompt friendly format.
        """
        return {
            "request": self.request.to_prompt_message_string(),
            "response": self.response.to_prompt_message_string(),
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

    def to_dict(self):
        return asdict(self)
    
    def create_turn(self, request: Message, response: Message) -> Turn:
        turn = Turn(uuid=str(uuid4()), request=request, response=response)
        self.turns.append(turn)
        self.last_active = response.timestamp
        return turn


def start_new_conversation(host: str, host_is_bot: bool, guest: str, guest_is_bot: bool) -> Conversation:
    conversation = Conversation(
        uuid=str(uuid4()),
        description= f"{host}-{guest}",
        created_at=datetime.now().strftime('%Y-%m-%d @ %H:%M'),
        last_active=datetime.now().strftime('%Y-%m-%d @ %H:%M'),
        host=host,
        host_is_bot=host_is_bot,
        guest=guest,
        guest_is_bot=guest_is_bot,
        turns=[]
    )
    print(f"New conversation {conversation.uuid} started!")
    return conversation


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
                history.append(turn.request.to_prompt_message_string())
                history.append(turn.response.to_prompt_message_string())
            return history
        return list(self.cache)
