from src.instructions import ModelInstructions


todos = [
    "Write mode prompts",
    "Get milk"
]

options = [
    "1 - To Prompt Script",
    "2 - To Prompt Script Md"
]

#print(f"Assistant Name: {instructions.name}")
#print(f"\nAssistant Intro: {instructions.assistant_intro}")

selected_option = input(f"Select an option:\n" + "\n".join(options) + "\n")

if selected_option == "1":
    instructions = ModelInstructions(method="load", assistant_name="juliet")
    prompt = instructions.to_prompt_script(user_request="Hello, are you there?")
    print(f"\nPrompt:\n{prompt}\n")
    
if selected_option == "2":
    instructions = ModelInstructions(method="load", assistant_name="juliet")
    md_prompt = instructions.to_prompt_script_md(user_request="Hello, are you there?", todos=todos)
    print(f"\nMd Prompt:\n{md_prompt}\n")





