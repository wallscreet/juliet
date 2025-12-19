import sys
from context import ChromaMemoryAdapter, ConversationManager, YamlMemoryAdapter
from instructions import ModelInstructions
from clients import LLMClient, XAIClient, IsoClient, OllamaClient
from context import Conversation, Message
from uuid import uuid4
import os
import yaml

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Markdown, TextArea, Static
from textual.containers import Container, VerticalScroll
from textual.binding import Binding
from rich.panel import Panel


class JulietChat(App):

    CSS = """
    #history {
        border: round orange;
        padding: 1;
    }

    #title {
        text-align: center;
        color: magenta;
        height: auto;
        padding: 1;
    }

    .message {
        margin: 1 0;
    }

    #user_input {
        border: round blue;
        margin-top: 0;
        height: 5;
    }
    """

    BINDINGS = [
        Binding("alt+enter", "send_message", "Send Message", show=True),
    ]

    def __init__(self, 
                 assistant_name: str, 
                 username: str, 
                 llm_client: LLMClient
        ):
        super().__init__()
        self.prompt_debug: bool = False
        self.token_counter: bool = True
        # Collect and normalize inputs
        self.assistant_name = assistant_name.strip().lower()
        self.username = username.strip().lower()

        # Build paths
        self.conversation_id = "42"
        self.chroma_persist_dir = f'isos/{self.assistant_name}/users/{self.username}/chroma_store'
        self.fact_store_path = f'isos/{self.assistant_name}/users/{self.username}/facts.yaml'
        self.conversations_path = f'isos/{self.assistant_name}/users/{self.username}/conversations.yaml'
        
        # Ensure full directory structure exists
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

        # 4. Touch (create if not exists) the YAML files
        for filepath in (self.conversations_path, self.fact_store_path):
            if not os.path.exists(filepath):
                # Create empty valid YAML file
                with open(filepath, 'w') as f:
                    yaml.safe_dump([], f)  # empty list is safe for both conversations and facts

        # Load Adapters - chroma, yaml and conversation manager from context.py
        # ChromaMemoryAdapter (db address)
        self.chroma_adapter = ChromaMemoryAdapter(persist_dir=self.chroma_persist_dir)
        # YamlMemoryAdapter (conversations file address)
        self.yaml_adapter = YamlMemoryAdapter(filepath=self.conversations_path)
        # ConversationManager (initialized yaml adapter)
        self.manager = ConversationManager(adapter=self.yaml_adapter)

        # Instructions file
        self.instructions = ModelInstructions(
                method="load", 
                assistant_name=self.assistant_name
        )
        self.llm_client = llm_client
        self.iso_client = IsoClient(
                llm_client=self.llm_client, 
                instructions=self.instructions, 
                chroma_adapter=self.chroma_adapter, 
                chroma_persist_dir=self.chroma_persist_dir, 
                fact_store_path=self.fact_store_path
        )

        # load or new conversation
        #choice = input("new or load? ").lower()
        choice = "load"
        if choice == "new":
            self.conversation = Conversation.start_new(
                host=self.instructions.name,
                host_is_bot=True,
                guest=self.username,
                guest_is_bot=False,
                uuid_override=self.conversation_id,
            )
            self.yaml_adapter.save_conversation(self.conversation)
        else:
            self.conversation = self.manager.get_or_start(
                conversation_id=self.conversation_id,
                host=self.instructions.name,
                host_is_bot=True,
                guest=self.username,
                guest_is_bot=False,
            )
            for turn in self.conversation.turns[-self.iso_client.message_cache.capacity:]:
                self.iso_client.message_cache.add_turn(turn=turn)

        #self.conversation_id = conversation_id
        print(f"Welcome, {self.username}")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        #yield Static(Panel.fit("[bold magenta]JulietChat[/]", border_style="cyan"), id="title")
        yield Container(
            VerticalScroll(id="history"),
            TextArea(placeholder="Type your message here... (Ctrl+Enter to send)", id="user_input"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.history = self.query_one("#history", VerticalScroll)
        self.user_input = self.query_one("#user_input", TextArea)
        self.user_input.focus()

        for turn in self.conversation.turns:
            self._add_to_history(f"**{turn.request.speaker}:**\n{turn.request.content}\n")
            self._add_to_history(f"**{turn.response.speaker}:**\n{turn.response.content}\n\n---")

    def _add_to_history(self, text: str) -> None:
        self.history.mount(Markdown(text, classes="message"))
        self.history.scroll_end(animate=True)

    def action_send_message(self) -> None:
        """Triggered by Alt+Enter."""
        user_input = self.user_input.text.strip()
        if not user_input:
            return

        if user_input == "/debug":
            # Toggle the debug flag
            self.prompt_debug = not self.prompt_debug
            status = "ON" if self.prompt_debug else "OFF"
            self._add_to_history(f"**System:** Debug prompt display is now **{status}**")
            self.user_input.text = ""  # Clear the input
            return
        
        if user_input == "/tokens":
            # Toggle the debug flag
            self.token_counter = not self.token_counter
            status = "ON" if self.token_counter else "OFF"
            self._add_to_history(f"**System:** Token Counter display is now **{status}**")
            self.user_input.text = ""  # Clear the input
            return

        self._add_to_history(f"**{self.username}:**\n{user_input}")
        self.user_input.text = ""

        response, prompt_messages, usage = self.iso_client.generate_response_with_tools(
            model="grok-4-1-fast-non-reasoning",
            #model="dolphin2.2-mistral",
            user_input=user_input,
        )
        
        # print full prompt to message history
        if self.prompt_debug:
            self._add_to_history(f"**Prompt Messages:**\n{prompt_messages}\n")
           # debug_str = "Full Prompt Messages (raw):\n\n"
           # for msg in prompt_messages:
               # if isinstance(msg, dict):
                   # role = msg.get('role', 'unknown')
                   # content = msg.get('content', '')
               # else:  # ChatCompletionMessage
                   # role = msg.role
                   # content = msg.content or str(msg.tool_calls) if msg.tool_calls else ''
        
               # debug_str += f"Role: {role}\nContent:\n{content}\n{'-'*50}\n"
    
           # self._add_to_history(f"**Prompt Debug:**\n{debug_str}")
        
        request_message = Message(
            uuid=str(uuid4()),
            role="user",
            speaker=f"{self.username}",
            content=user_input,
        )

        response_message = Message(
            uuid=str(uuid4()),
            role="assistant",
            speaker=self.instructions.name,
            content=response,
        )

        turn = self.manager.add_turn(self.conversation, request_message, response_message)

        self.iso_client.message_cache.add_turn(turn=turn)
        self.chroma_adapter.store_turn(conversation_id=self.conversation_id, turn=turn)

        self._add_to_history(f"**{self.instructions.name}:**\n{response}\n\n")

        # print token usage to message history
        if self.token_counter:
            self._add_to_history(f"**Token Usage:**\n{usage}\n\n---")
         
        

if __name__ == "__main__":
    try:
        llm_client = XAIClient()
        assistant_name = input("Enter assistant name: ")
        username = input("Enter your username: ")
        
        app = JulietChat(assistant_name=assistant_name, 
                         username=username, 
                         llm_client=llm_client
        )
        
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)
