from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
import os
from uuid import uuid4
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from instructions import ModelInstructions
from messages import MessageCache
from context import ChromaMemoryAdapter, message_cache_format_to_prompt


load_dotenv()


class LLMClient(ABC):
    """
    Abstract base class for all LLM clients. Defines the standard interface so different SDKs can be swapped out without changing the chat client code.
    """

    @abstractmethod
    def get_response(self, model: str, messages: List[dict], temp: float = 0.3, **kwargs) -> str:
        """
        Get a response from the model as plain text.
        """
        pass

    @abstractmethod
    def get_structured_response(self, model: str, response_format: type[BaseModel], content: str, **kwargs) -> BaseModel:
        """
        Get a structured/parsed response from the model.
        """
        pass


@dataclass
class XAIClient(LLMClient):
    api_key: str = os.getenv("XAI_API_KEY")
    base_url: str = "https://api.x.ai/v1"

    def __post_init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=3600,
        )
    
    def get_response(self, model: str, messages: list = None):
        """
        Get a response from the Grok AI model.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}")
    
    def get_structured_response(self, model: str, response_format: type[BaseModel] = None, content: str = None) -> BaseModel:
        """
        Get a structured output response from the Grok AI api.
        """
        messages = [
            {
                "role": "system",
                "content": "Extract structured information from the content."
            },
            {
                "role": "user",
                "content": content
            }
        ]

        try:
            completion = self.client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
            )
            return completion.choices[0].message.parsed
        
        except Exception as e:
            print(f"Error: {e}")


@dataclass
class OllamaClient(LLMClient):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"

    def __post_init__(self):
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_response(self, model: str, messages: List[dict], **kwargs) -> str:
        try:
            completion = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in OllamaClient.get_response: {e}")

    def get_structured_response(self, model: str, response_format: type[BaseModel], content: str, **kwargs) -> BaseModel:
        try:
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": content}],
                format=response_format.model_json_schema(),
                **kwargs
            )
            return response_format.model_validate_json(response.message.content)
        except Exception as e:
            print(f"Error in OllamaClient.get_structured_response: {e}")


@dataclass
class OpenAIClient(LLMClient):
    api_key: str = os.getenv("OPENAI_API_KEY")

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, model: str, messages: List[dict], **kwargs) -> str:
        """
        Get a plain text response from OpenAI models (ChatGPT, GPT-4, etc).
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAIClient.get_response: {e}")

    def get_structured_response(self, model: str, response_format: type[BaseModel], content: str, **kwargs) -> BaseModel:
        """
        Get structured response using Pydantic schema validation.
        """
        messages = [
            {"role": "system", "content": "Extract structured information from the content."},
            {"role": "user", "content": content}
        ]
        try:
            completion = self.client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Error in OpenAIClient.get_structured_response: {e}")


class IsoClient:
    """
    Iso agent class that combines instructions, parameters, LLM client, and memory. Responsible for building prompts dynamically, handling context, and generating responses.
    """

    def __init__(self, llm_client: LLMClient, instructions: ModelInstructions, cache_capacity: int = 20):
        self.llm_client = llm_client
        self.instructions = instructions
        self.message_cache = MessageCache(capacity=cache_capacity)
        self.memory = ChromaMemoryAdapter()
    
    def build_prompt(self, user_input: str):
        # chat history
        chat_history = self.message_cache.get_chat_history()
        chat_history_formatted = message_cache_format_to_prompt(message_history=chat_history)
        #print(f"Message Cache Formatted: {chat_history_formatted}")
        
        # memory context
        memory_context = self.memory.retrieve(collection_name="memory", query=user_input)
        #print(f"Memory Context: {memory_context}")

        # knowledge context
        knowledge_context = self.memory.retrieve(collection_name="knowledge", query=user_input)
        #print(f"Knowledge Context: {knowledge_context}")
        
        # return instructions to prompt script
        messages = self.instructions.to_prompt_script(mem_context=memory_context, knowledge_context=knowledge_context, chat_history=chat_history_formatted, user_request=user_input)
        
        return messages
    
    def generate_response(self, model: str, user_input: str):
        # create request and response messages and add to message cache
        prompt = self.build_prompt(user_input=user_input)
        print(f"\n===== PROMPT =====\n{prompt}\n\n")
        return self.llm_client.get_response(model=model, messages=prompt)



class ChatClient(ABC):
    @abstractmethod
    def chat_loop(self):
        pass


class CliChatClient(ChatClient):
    def __init__(self):
        pass

    def chat(self):
        pass
