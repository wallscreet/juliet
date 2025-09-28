import sys
from context import ChromaMemoryAdapter, ConversationManager, YamlMemoryAdapter
from instructions import ModelInstructions
from clients import XAIClient, IsoClient
from context import Conversation, Message
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextArea
from textual.containers import Container


class JulietChat(App):
    CSS = """
    #history {
        border: round orange;
        padding: 1;
    }

    #user_input {
        border: round blue;
        margin-top: 0;
    }
    """

    def __init__(self):
        super().__init__()
        self.chroma_adapter = ChromaMemoryAdapter(persist_dir="./chroma_store")
        self.yaml_adapter = YamlMemoryAdapter(filepath="isos/juliet/conversations.yaml")
        self.manager = ConversationManager(adapter=self.yaml_adapter)

        self.instructions = ModelInstructions(method="load", assistant_name="juliet")
        self.llm_client = XAIClient()
        self.iso_client = IsoClient(llm_client=self.llm_client, instructions=self.instructions)

        # load or new conversation
        conversation_id = "42"
        choice = input("new or load? ").lower()
        if choice == "new":
            self.conversation = Conversation.start_new(
                host=self.instructions.name,
                host_is_bot=True,
                guest="Wallscreet",
                guest_is_bot=False,
                uuid_override=conversation_id,
            )
            self.yaml_adapter.save_conversation(self.conversation)
        else:
            self.conversation = self.manager.get_or_start(
                conversation_id=conversation_id,
                host=self.instructions.name,
                host_is_bot=True,
                guest="Wallscreet",
                guest_is_bot=False,
            )
            for turn in self.conversation.turns[-self.iso_client.message_cache.capacity:]:
                self.iso_client.message_cache.add_turn(turn=turn)

        self.conversation_id = conversation_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            TextArea(id="history", read_only=True),
            Input(placeholder="Type your message here...", id="user_input"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.history = self.query_one("#history", TextArea)
        self.user_input = self.query_one("#user_input", Input)
        self.user_input.focus()

        for turn in self.conversation.turns:
            self._add_to_history(f"\n{turn.request.speaker}:\n{turn.request.content}\n\n")
            self._add_to_history(f"{turn.response.speaker}:\n{turn.response.content}\n\n-----------------------------------------------------\n")

    def _add_to_history(self, text: str) -> None:
        self.history.text += text
        self.history.cursor_location = (self.history.document.line_count - 1, 0)
        self.history.scroll_to(self.history.cursor_location, animate=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        self._add_to_history(f"\nWallscreet:\n{user_input}\n\n")
        self.user_input.value = ""

        response = self.iso_client.generate_response(
            model="grok-4-fast-non-reasoning",
            user_input=user_input,
        )

        request_message = Message(
            uuid=str(uuid4()),
            role="user",
            speaker="Wallscreet",
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

        self._add_to_history(f"{self.instructions.name}:\n{response}\n\n-----------------------------------------------------\n")


if __name__ == "__main__":
    try:
        app = JulietChat()
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)
