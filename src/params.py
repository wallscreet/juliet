from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class ParamsConfig:
    """
    Iso configuration dataclass for tweaking completion parameters. More are available through Ollama's API, I will build this out to cover it all eventually. Parameter definitions from Ollama and their defaults values are given in params. Class field defaults are values that I have found to work well for my use cases.

    :param temperature: The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
    :param num_ctx: Sets the size of the context window used to generate the next token. (Default: 4096)
    :param num_gpu: The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. (Default: 50)
    :param num_thread: Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). (Default: 8, I run 16 on a core i9)
    :param top_k: Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    :param top_p: Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    :param num_predict: The number of tokens to generate. (Default: 128)
    :param seed: The seed to use for random number generation. (Default: 0)
    :param mirostat: Enables the Mirostat algorithm. (Default: 0)
    :param mirostat_eta: The learning rate for the Mirostat algorithm. (Default: 0.1)
    :param mirostat_tau: The temperature for the Mirostat algorithm. (Default: 5.0)
    :param repeat_last_n: The number of tokens to repeat at the end of the context. (Default: 64)
    :param completions_url: The URL of the Ollama API endpoint.
    :param completion_headers: The headers to send with the request to the Ollama API.
    :param start_token: The token to use to start the prompt.
    :param end_token: The token to use to end the prompt.
    :param tfs_z: The number of tokens to use for the TFS-Z algorithm. (Default: 0)
    :creates: Param config object for the iso.
    """
    temperature: float = None
    num_ctx: int = None
    num_gpu: int = None
    num_thread: int = None
    top_k: int = None
    top_p: float = None
    num_predict: int = None
    seed: int = None
    mirostat: int = None
    mirostat_eta: float = None
    mirostat_tau: float = None
    repeat_last_n: int = None
    tfs_z: int = None
    assistant_name: str = None

    def __init__(self, method: str, assistant_name: str) -> None:
        """
        Model Params Config init takes a method param as ['create', 'load'] to determine if the instructions should be loaded from a yaml file or created from the CLI.

        :param method: The method to use to create the instructions.
        """
        self.assistant_name = assistant_name
        if method == 'load':
            self.load_from_yaml()
            print(f"Loaded param config for {assistant_name}")
            print(asdict(self))
        elif method == 'create':
            self.load_defaults_from_yaml()
            print("Creating new completion parameters configuration...")
            customize = input("Would you like to customize the model params? (y/n): ").strip()
            if customize == 'y':
                self.update_model_params()
                print(asdict(self))
            else:
                print("Using default completion parameters. You can customize these later.")
                print(asdict(self))
        else:
            print("Error: Invalid method. Please use 'create' or 'load'.")
    
    def to_dict(self) -> dict:
        """
        Export config class to a base dict

        :returns: Base dictionary for the config class.
        """
        return asdict(self)
    
    def update_model_params(self) -> None:
        """
        Iterate through the config and update the values or keep current.
        """
        params = self.to_dict()
        for key, value in params.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                setattr(self, key, new_value)
        self.save_to_yaml(self.assistant_name)
    
    def print_config(self) -> None:
        """
        Print the config to the terminal.

        :returns: Prints a pre-defined config string to the terminal.
        """
        print(f"Iso Configuration:\n{self.to_dict()}")
    
    def cli_create_config(self) -> None:
        """
        Create a config from the CLI.
        """
        params = self.to_dict()
        for key, value in params.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                params[key] = new_value
    
    def load_from_yaml(self) -> None:
        """
        Load the iso instructions config from a yaml file.
        """
        params_config = Path(f"isos/{self.assistant_name.lower()}/params_config.yaml")
        if params_config.exists():
            with params_config.open('r') as file:
                completion_params = yaml.safe_load(file)
                for key, value in completion_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def load_defaults_from_yaml(self) -> None:
        """
        Load the iso instructions config from a yaml file.
        """
        params_config = Path(f"iso-template/params_config.yaml")
        if params_config.exists():
            with params_config.open('r') as file:
                completion_params = yaml.safe_load(file)
                for key, value in completion_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save_to_yaml(self) -> None:
        """
        Save the iso config to a yaml file.

        :returns: Saves the iso config to a yaml file.
        """
        data = self.to_dict()
        with open(f"/isos/{self.assistant_name.lower()}/params_config.yaml", "w") as f:
            yaml.safe_dump(data, f)
