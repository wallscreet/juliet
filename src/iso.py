from datetime import datetime
import re
from string import Template
from typing import Optional
from uuid import uuid4
from clients import LLMClient
from instructions import ModelInstructions
from messages import Message, MessageCache, Turn
from params import ParamsConfig
from context import MemoryAdapter, ChromaMemoryAdapter, YamlMemoryAdapter


class Iso:
    """
    Iso agent class that combines instructions, parameters, LLM client, and memory.
    Responsible for building prompts dynamically, handling context, and sending messages.
    """

    def __init__(self, llm_client: LLMClient, instructions, params_config, short_term_memory_capacity: int = 20, memory_adapter: Optional[MemoryAdapter] = None):
        self.llm_client = llm_client
        self.instructions = instructions
        self.params_config = params_config
        self.name = self.instructions.name
        self.message_cache = MessageCache(capacity=short_term_memory_capacity)
        self.yaml_adapter = YamlMemoryAdapter()
        self.chroma_adapter = ChromaMemoryAdapter()
        self.last_response: Optional[str] = None

    