import sys
from context import ChromaMemoryAdapter, YamlMemoryAdapter
from instructions import ModelInstructions
from clients import XAIClient, IsoClient
from context import Conversation, Message
from uuid import uuid4


def main():
    chroma_adapter = ChromaMemoryAdapter(persist_dir="./chroma_store")
    yaml_adapter = YamlMemoryAdapter(filepath="conversations.yaml")
    instructions = ModelInstructions(method="load", assistant_name="clappy")
    #print(f"\nInstructions: {instructions.to_dict()}\n")
    llm_client = XAIClient()
    iso_client = IsoClient(llm_client=llm_client, instructions=instructions)

    # TODO: load or new conversation
    choice = input("new or load? ").lower()
    
    if choice == "new":
        conversation = Conversation.start_new(host=instructions.name, host_is_bot=True, guest="Wallscreet", guest_is_bot=False, uuid_override="42")
    
    # TODO: Fix load from yaml
    elif choice == "load":
        conversation = yaml_adapter.load_conversation_by_id(conversation_id="42")

    
    # TODO: while true loop
    while True:
        user_input = input("User Input: ")
        
        response = iso_client.generate_response(model="grok-4-fast-non-reasoning", user_input=user_input)

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
        
        turn = conversation.create_turn(request=request_message, response=response_message, conversation_id="42")
        
        conversations_filepath = instructions.conversations_filepath
        conversation.save_to_yaml(yaml_path=conversations_filepath)

        iso_client.message_cache.add_turn(turn=turn)

        print(f"Response: {response}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
