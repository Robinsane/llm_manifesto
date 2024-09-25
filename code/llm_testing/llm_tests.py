import random
import threading
import time
from typing import Mapping

from ollama import Client

SERVER_IP = "192.168.0.17"
OLLAMA_CLIENT = Client(host=f"http://{SERVER_IP}:11434")
TEST_SYSTEM_PROMPT = "You are a helpful assistant providing SHORT, correct and clear answers."


class Model:
    def __init__(self, model: str, name: str, size: int, parameter_size: str, quant: str):
        # TODO name aya quant shows F16, probably wrong? ... -> inform Ollama
        self.model = model
        self.name = name
        self.size_gb = round(size / (1024 ** 3), 1)
        self.parameter_size = parameter_size
        self.quant = quant

        if self.name != self.model:
            print(f"Model and name differ, find out what the difference is and why!\n {self.name} | {self.model}")

    @property
    def base_name(self) -> str:
        """get name name w/o tag."""
        return self.name.split(":")[0]

    @property
    def tag(self):
        """Get the tag of this model."""
        return self.name.split(":")[1]


class Generation:
    """Class representing a generation and it's important datapoints."""
    def __init__(self, system_prompt: str, prompt:str, response: Mapping[str, any]):
        self.model = response["model"]
        self.system_prompt = system_prompt
        self.prompt = prompt
        # speeds are in T/s:
        self.duration = round(response["total_duration"] * (10 ** -9), 1)
        self.eval_speed = round(response["prompt_eval_count"] / (response["prompt_eval_duration"] * (10 ** -9)), 1)
        self.response_speed = round(response["eval_count"] / (response["eval_duration"] * (10 ** -9)), 1)
        self.content = response["response"]

    @property
    def csv_row(self) -> str:
        splits = self.model.split(":")
        base_name = splits[0]
        tag = splits[1]
        formatted_system_prompt = self.system_prompt.replace(";", "").replace("\n", "<br>")
        formatted_prompt = self.prompt.replace(";", "").replace("\n", "<br>")
        formatted_content = self.content.replace(";", "").replace("\n", "<br>")
        return f"{base_name}; {tag}; {self.duration}; {self.eval_speed}; {self.response_speed}; {formatted_system_prompt}; {formatted_prompt}; {formatted_content}\n"


def get_all_models() -> list[Model]:
    """Get all available models available in the OLLAMA_CLIENT. (sorted from small -> big)"""
    available_models = []
    for model_dict in OLLAMA_CLIENT.list()["models"]:
        model = Model(
            model_dict["model"],
            model_dict["name"],
            model_dict["size"],
            model_dict["details"]["parameter_size"],
            model_dict["details"]["quantization_level"]
        )
        available_models.append(model)

    available_models.sort(key=lambda m: m.size_gb)

    return available_models


def get_general_models() -> list[Model]:
    """Get all general (non-coding) models available in the OLLAMA_CLIENT."""

    return [m for m in get_all_models() if "cod" not in m.name]


def get_coding_models() -> list[Model]:
    """Get all coding models available in the OLLAMA_CLIENT."""

    return [m for m in get_all_models() if "cod" in m.name]


def load_models(models: list[Model], wait: bool = True):
    """Load all models into RAM memory."""
    threads = []
    for model in models:
        # If an empty prompt is provided, the model will start loading into memory
        thread = threading.Thread(target=OLLAMA_CLIENT.generate, args=(model.name,))
        threads.append(thread)
        thread.start()

    if wait:
        print("Waiting while models are being loaded in RAM...")
        timestamp = time.time()
        for thread in threads:
            thread.join()
        print(f"All models are loaded in RAM! (took {round(time.time() - timestamp)} seconds)")


def test_models(models: list[Model], output_file: str = "output/speed_tests.csv"):
    """Sequentially prompt each given model + output the speeds and answers to output/speed_tests_1.csv ."""
    print("--- Testing all available models ---")

    prompt = generate_random_question()
    print(f"Prompt = {prompt}")
    for model in models:
        print(f"Prompting {model.base_name} ({model.tag}) ({model.quant}) ...")
        # Note: using chat because generate doesn't nicely give the eval times etc.
        response = OLLAMA_CLIENT.generate(
            model=model.name,
            system=TEST_SYSTEM_PROMPT,
            prompt=prompt,
            stream=False,
            # format="json",  # "if not json, the model may generate large amounts whitespace!"
            # options=[],  # model parameters listed in the documentation for the Modelfile
            keep_alive=-1,  # already set /w env variable, but w/e
        )

        answer = Generation(TEST_SYSTEM_PROMPT, prompt, response)
        with open(output_file, "a", encoding="utf-8") as speed_csv:
            speed_csv.write(answer.csv_row)

    # output an extra whiteline to indicate a finished run of all models
    with open(output_file, "a") as speed_csv:
        speed_csv.write("\n")


def generate_random_question() -> str:
    """Generate a random question with the smallest available ollama model."""
    smallest_model = get_general_models()[0]  # for me this will likely always be phi 3.5
    response = OLLAMA_CLIENT.generate(
        model=smallest_model.name,
        prompt="Generate a short, one sentence question. OUTPUT ONLY THIS 1 SENTENCE LONG QUESTION.",
        system="You are a random question generator. "
               "You generate short question across diverse topics."
               "Be unpredictable and avoid repeating themes or words."
               "OUTPUT ONLY THIS 1 SENTENCE LONG QUESTION",
        options={
            "seed": random.randint(-2_000_000_000, 2_000_000_000),
            "temperature": 5,  # higher == more random output (0-5?)
            "top-p": 0.98,  # nucleus sampling (max 1); higher means choose next word from bigger pool
            "top_k": 500,  # limit the number of next-word candidates
            "frequency_penalty": 1.2,  # impacts repetition of tokens within the same output
            "presence_penalty": 0.8,  # not impactful unless chaining / extending responses
        },
    )

    return response["response"]


def generate_random_coding_question() -> str:
    """Generate a random question with the smallest available ollama model."""
    smallest_model = get_general_models()[0]  # for me this will likely always be phi 3.5
    response = OLLAMA_CLIENT.generate(
        model=smallest_model.name,
        prompt="Generate a short, one sentence, coding-related question. OUTPUT ONLY THIS 1 SENTENCE LONG QUESTION.",
        system="You are a random question generator. "
               "You generate short question related to programming."
               "Be unpredictable and avoid repeating themes or words."
               "OUTPUT ONLY THIS 1 SENTENCE LONG QUESTION",
        options={
            "seed": random.randint(-2_000_000_000, 2_000_000_000),
            "temperature": 5,  # higher == more random output (0-5?)
            "top-p": 0.98,  # nucleus sampling (max 1); higher means choose next word from bigger pool
            "top_k": 500,  # limit the number of next-word candidates
            "frequency_penalty": 1.2,  # impacts repetition of tokens within the same output
            "presence_penalty": 0.8,  # not impactful unless chaining / extending responses
        },
    )

    return response["response"]


# if __name__ == "__main__":
#     # temp testing section:

