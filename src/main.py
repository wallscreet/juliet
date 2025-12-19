import sys
from context import ChromaMemoryAdapter, ConversationManager, YamlMemoryAdapter
from instructions import ModelInstructions
from clients import XAIClient, IsoClient
from context import Conversation, Message
from uuid import uuid4


def main():
    chroma_adapter = ChromaMemoryAdapter(persist_dir="./chroma_store")
    yaml_adapter = YamlMemoryAdapter(filepath="isos/juliet/conversations.yaml")
    manager = ConversationManager(adapter=yaml_adapter)

    instructions = ModelInstructions(method="load", assistant_name="juliet")
    llm_client = XAIClient()
    iso_client = IsoClient(llm_client=llm_client, instructions=instructions)

    choice = input("new or load? ").lower()
    conversation_id = "42"

    if choice == "new":
        # force overwrite with new conversation
        conversation = Conversation.start_new(
            host=instructions.name,
            host_is_bot=True,
            guest="Wallscreet",
            guest_is_bot=False,
            uuid_override=conversation_id,
        )
        yaml_adapter.save_conversation(conversation)
    else:
        conversation = manager.get_or_start(
            conversation_id=conversation_id,
            host=instructions.name,
            host_is_bot=True,
            guest="Wallscreet",
            guest_is_bot=False,
        )
        #print(f"Last Conversation Turn:\n{conversation.turns[-1]}")
        for turn in conversation.turns[-iso_client.message_cache.capacity:]:
            iso_client.message_cache.add_turn(turn=turn)


    # ==== Chat loop ====
    while True:
        user_input = input("\nUser Input: ")
        
        response = iso_client.generate_response(
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
            speaker=instructions.name,
            content=response,
        )

        turn = manager.add_turn(conversation, request_message, response_message)

        iso_client.message_cache.add_turn(turn=turn)
        chroma_adapter.store_turn(conversation_id=conversation_id, turn=turn)

        print(f"\nResponse: {response}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
