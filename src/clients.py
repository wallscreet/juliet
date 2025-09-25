from dataclasses import dataclass
import traceback
from typing import List, Any, Optional
from abc import ABC, abstractmethod
import os
from uuid import uuid4
from pydantic import BaseModel
from openai import OpenAI
from context import MemoryAdapter
from messages import Message, MessageCache, start_new_conversation
from dotenv import load_dotenv

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


class ChatClient(ABC):
    @abstractmethod
    def chat_loop(self):
        pass


class CliChatClient(ChatClient):
    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        host: str = "Assistant",
        guest: str = "User",
        memory_cache: Optional[MessageCache] = None,
        memory_adapter: Optional[MemoryAdapter] = None,
    ):
        """
        from GPT-5, CLI-based chat client with pluggable memory. I don't like this because I need to have the YAML adapter AND the ChromaAdapter plugged in.

        :param llm_client: Implementation of LLMClient.
        :param model: Model name to use.
        :param host: The Iso's name.
        :param guest: The user's name.
        :param memory_cache: Short-term memory (deque).
        :param memory_adapter: Long-term memory backend.
        """
        self.llm_client = llm_client
        self.model = model
        self.memory_cache = memory_cache or MessageCache(capacity=10)
        self.memory_adapter = memory_adapter

        self.conversation = start_new_conversation(
            host=host,
            host_is_bot=True,
            guest=guest,
            guest_is_bot=False,
        )

    def _create_message(self, role: str, speaker: str, content: str) -> Message:
        return Message(
            uuid=str(uuid4()),
            role=role,
            speaker=speaker,
            content=content,
        )

    def chat_loop(self):
        """
        Interactive REPL loop for chatting.
        """
        print("Welcome to JulietChat! Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_input = input(f"{self.conversation.guest}: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                # --- Build User Message ---
                user_message = self._create_message(
                    role="user",
                    speaker=self.conversation.guest,
                    content=user_input,
                )

                # Collect context from short-term memory
                messages_for_llm = [{"role": "system", "content": "You are a helpful AI assistant."}]
                for turn in self.memory_cache.get_chat_history(as_strings=False):
                    messages_for_llm.append({"role": "user", "content": turn.request.content})
                    messages_for_llm.append({"role": "assistant", "content": turn.response.content})
                messages_for_llm.append({"role": "user", "content": user_input})

                # --- Get AI Response ---
                response_text = self.llm_client.get_response(
                    model=self.model,
                    messages=messages_for_llm,
                )

                iso_message = self._create_message(
                    role="assistant",
                    speaker=self.conversation.host,
                    content=response_text,
                )

                # --- Record Turn ---
                turn = self.conversation.create_turn(request=user_message, response=iso_message)
                self.memory_cache.add_turn(turn)

                if self.memory_adapter:
                    self.memory_adapter.store_turn(self.conversation.uuid, turn)

                # --- Print AI Response ---
                print(f"{self.conversation.host}: {response_text}\n")

            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print("Error in chat loop:", e)
                traceback.print_exc()
                break
