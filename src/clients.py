from dataclasses import dataclass
import json
from typing import Any, Dict, List
from abc import ABC, abstractmethod
import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from instructions import ModelInstructions
from messages import MessageCache
from context import ChromaMemoryAdapter
from fact_store import Fact, FactStore
from files_handler import FileCreateRequest, FileEditRequest, FileDeleteRequest, FileHandler


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

    @abstractmethod
    def get_response_with_tools(self, model: str, messages: List[dict], temp: float = 0.3, **kwargs) -> str:
        """
        Get a response with the tool pipeline included.
        """
        pass

@dataclass
class XAIClient(LLMClient):
    """
    ok TESTED
    """
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
        Get a response from an xAi model.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}")
    
    def get_structured_response(self, model: str, response_format: type[BaseModel] = None, content: str = None) -> BaseModel:
        """
        Get a structured output response from the xAi api.
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
    
    def get_response_with_tools(self, model: str, messages: list, tools: list = None):
        """
        Get a response from the xAi api with tools pipeline
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message, completion.usage
        except Exception as e:
            print(f"Error: {e}")


@dataclass
class OllamaClient(LLMClient):
    """
    ok TESTED
    The ollama docs say I can use the openai sdk and use the v1 endpoint.
    """
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"

    def __post_init__(self):
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_response(self, model: str, messages: list = None):
        """
        Get a response from a local Ollama model.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}")
    
    def get_structured_response(self, model: str, response_format: type[BaseModel] = None, content: str = None) -> BaseModel:
        """
        Get a structured output response from the local Ollama api.
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
    
    def get_response_with_tools(self, model: str, messages: list, tools: list = None):
        """
        Get a response from a local Ollama model with tools pipeline
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message, completion.usage
        except Exception as e:
            print(f"Error: {e}")


@dataclass
class OpenAIClient(LLMClient):
    """
    NOT TESTED
    """
    api_key: str = os.getenv("OPENAI_API_KEY")

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, model: str, messages: list = None):
        """
        Get a response from the OpenAI api.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}")
    
    def get_structured_response(self, model: str, response_format: type[BaseModel] = None, content: str = None) -> BaseModel:
        """
        Get a structured output response from the OpenAI api.
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
    
    def get_response_with_tools(self, model: str, messages: list, tools: list = None):
        """
        Get a response from the OpenAI api with tools pipeline
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            print(f"Tokens Usage:\n{completion.usage}\n")
            return completion.choices[0].message, completion.usage
        except Exception as e:
            print(f"Error: {e}")


class IsoClient:
    """
    Iso agent class that combines instructions, parameters, LLM client, and memory. 
    Responsible for building prompts dynamically, handling context, and generating responses.
    """

    def __init__(self, 
                 llm_client: LLMClient, 
                 chroma_adapter: ChromaMemoryAdapter, 
                 chroma_persist_dir: str, 
                 instructions: ModelInstructions, 
                 fact_store_path: str, 
                 cache_capacity: int = 20
        ):
        
        self.llm_client = llm_client
        self.instructions = instructions
        self.message_cache = MessageCache(capacity=cache_capacity)
        self.memory = chroma_adapter
        self.fact_store = FactStore(
                fact_store_path=fact_store_path, 
                chroma_persist_dir=chroma_persist_dir
        )
        self.file_handler = FileHandler()
        self._tools = []
        self.use_docs = False

        self.register_tool(
            name="add_fact",
            description="Add a new fact as a subject-predicate-object triple.",
            parameters=Fact.model_json_schema()
        )
        self.register_tool(
            name="create_file",
            description="Create a new file with given filename and content",
            parameters=FileCreateRequest.model_json_schema()
        )
        self.register_tool(
            name="edit_file",
            description="Edit an existing file with new content",
            parameters=FileEditRequest.model_json_schema()
        )
        self.register_tool(
            name="delete_file",
            description="Delete a file by filename",
            parameters=FileDeleteRequest.model_json_schema()
        )

    def build_prompt(self, user_input: str):
        # chat history
        chat_history = self.message_cache.get_chat_history()
        #print(f"Chat History: {chat_history}")
        
        # memory context
        memory_context = self.memory.retrieve(collection_name="memory", query=user_input)
        memory_context_formatted = []
        for message in memory_context:
            memory_context_formatted.append(message.to_content_string())
        #print(f"Memory Context: {memory_context}")

        # knowledge context
        knowledge_context = self.memory.retrieve(collection_name="knowledge", query=user_input)
        #print(f"Knowledge Context: {knowledge_context}")

        # facts context
        facts_context = self.fact_store.retrieve_facts(query=user_input)
        facts_formatted = []
        for fact in facts_context:
            facts_formatted.append(fact.to_memory_string())
        #print(f"Facts Context: {facts_formatted}")

        # contents of workspace dir
        workspace_contents = self.file_handler.list_files()
        
        # return instructions to prompt script
        messages = self.instructions.to_prompt_script(
                facts_context=facts_formatted, 
                mem_context=memory_context_formatted, 
                knowledge_context=knowledge_context, 
                chat_history=chat_history, 
                user_request=user_input, 
                workspace_contents=workspace_contents
        )
        
        return messages
    
    def generate_response(self, model: str, user_input: str):
        prompt = self.build_prompt(user_input=user_input)
        return self.llm_client.get_response(model=model, messages=prompt)
    
    def register_tool(self, name: str, description: str, parameters: dict):
        """Register a new tool that the LLM can call."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self._tools.append(tool_spec)

    def get_tools(self):
        return self._tools
    
    def generate_response_with_tools(self, model: str, user_input: str):
        messages = self.build_prompt(user_input)
        tools = self.get_tools()
        
        while True:
            response, usage = self.llm_client.get_response_with_tools(model, messages, tools)
            
            if not response.tool_calls:
                return response.content, messages, usage
            
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if tool_name == "add_fact":
                    fact = Fact(**args)
                    self.fact_store.store_fact(fact)
                    result = {"status": "fact_added", "fact": args}

                elif tool_name == "create_file":
                    result = self.file_handler.create_file(args)

                elif tool_name == "edit_file":
                    result = self.file_handler.edit_file(args)

                elif tool_name == "delete_file":
                    result = self.file_handler.delete_file(args)

                else:
                    result = {"status": "unknown_tool", "tool": tool_name}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })
        

# The ais are pretty insistent on using these abstract base classes.
class ChatClient(ABC):
    @abstractmethod
    def chat_loop(self):
        pass


# TODO: Need a cli chat client pretty bad!
class CliChatClient(ChatClient):
    def __init__(self):
        pass

    def chat(self):
        pass
