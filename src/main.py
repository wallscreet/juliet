import sys
from context import ChromaMemoryAdapter, YamlMemoryAdapter
from instructions import ModelInstructions
from clients import XAIClient, IsoClient


def main():
    chroma_adapter = ChromaMemoryAdapter(persist_dir="./chroma_store")
    yaml_adapter = YamlMemoryAdapter(filepath="conversations.yaml")
    instructions = ModelInstructions(method="load", assistant_name="clappy")
    #print(f"\nInstructions: {instructions.to_dict()}\n")
    llm_client = XAIClient()
    iso = IsoClient(llm_client=llm_client, instructions=instructions)
    # TODO: load or new conversation
    # TODO: while true loop
    user_input = input("User Input: ")
    
    response = iso.generate_response(model="grok-4-fast-non-reasoning", user_input=user_input)

    print(f"Response: {response}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
