import random
import threading
import time
from pathlib import Path
from typing import Mapping

import polars as pl
from ollama import Client

SERVER_IP = "192.168.0.17"
OLLAMA_CLIENT = Client(host=f"http://{SERVER_IP}:11434")
TEST_SYSTEM_PROMPT = "You are a helpful assistant providing SHORT, correct and clear answers."
QUESTION_GENERATION_MODEL = "llama3.2:3b-instruct-q4_K_M"  # imo the fastest & best model to generate questions
HARDWARE_STR = "4x E7-4880 v2"  # CPU's / GPU the models will run on
GENERATIONS_FILE = Path("files/generations.parquet")

GENERATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


class Model:
    def __init__(self, model_dict: dict):
        """Create a custom model class based on the answer dict response from Ollama."""
        # TODO name aya quant shows F16, probably wrong? ... -> inform Ollama
        self.model = model_dict["model"]
        self.name = model_dict["name"]
        self.size_gb = round(model_dict["size"] / (1024 ** 3), 1)
        self.parameter_size = model_dict["details"]["parameter_size"]
        self.quant = model_dict["details"]["quantization_level"]

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
    """Class representing a generation."""
    def __init__(self, model: Model, system_prompt: str, prompt: str, response: Mapping[str, any], hardware_str: str = HARDWARE_STR):
        self.model = model
        self.hardware = hardware_str  # what the models are run on
        self.system_prompt = system_prompt
        self.prompt = prompt
        # note: speeds are in T/s:
        self.duration = round(response["total_duration"] * (10 ** -9), 1)
        self.eval_speed = round(response["prompt_eval_count"] / (response["prompt_eval_duration"] * (10 ** -9)), 1)
        self.response_speed = round(response["eval_count"] / (response["eval_duration"] * (10 ** -9)), 1)
        self.content = response["response"]

    @property
    def model_name(self) -> str:
        return self.model.name

    def to_dict(self) -> dict:
        """Convert the Generation object to a dictionary."""
        return {
            "model": self.model_name,
            "parameter_size": self.model.parameter_size,
            "size_gb": self.model.size_gb,
            "hardware": self.hardware,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt,
            "duration": self.duration,
            "eval_speed": self.eval_speed,
            "response_speed": self.response_speed,
            "content": self.content
        }


def get_all_models() -> list[Model]:
    """Get all available models available in the OLLAMA_CLIENT. (sorted from small -> big)"""
    available_models = []
    for model_dict in OLLAMA_CLIENT.list()["models"]:
        model = Model(model_dict)
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


def generate_random_question(model_name: str = None) -> str:
    """Generate a random question with the smallest available ollama model."""

    if model_name is None:
        model_name = get_general_models()[0].name  # if no model name is given, use the smallest one available

    response = OLLAMA_CLIENT.generate(
        model=model_name,
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


def generate_random_coding_question(model_name: str = None) -> str:
    """Generate a random question with the smallest available ollama model."""

    if model_name is None:
        model_name = get_general_models()[0].name  # if no model name is given, use the smallest one available

    response = OLLAMA_CLIENT.generate(
        model=model_name,
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


def save_generations(generations: list[Generation], parquet_output_path: Path | str, also_output_xlsx: bool):
    """Save or append given list of generations to a parquet file (optionally also to xlsx)."""

    generation_dicts = [gen.to_dict() for gen in generations]

    df = pl.DataFrame(generation_dicts)

    if type(parquet_output_path) == str:
        parquet_output_path = Path(parquet_output_path)
    if GENERATIONS_FILE.exists():
        existing_df = pl.read_parquet(parquet_output_path)
        df = pl.concat([existing_df, df])

    df.write_parquet(parquet_output_path)

    if also_output_xlsx:
        pandas_df = df.to_pandas()
        excel_file = parquet_output_path.parent / "generations.xlsx"
        pandas_df.to_excel(excel_file, index=False)


def test_models(models: list[Model], parquet_output_path: Path | str = GENERATIONS_FILE):
    """Prompt each given model. Once done the generations are saved/ appended to a parquet and xlsx file."""
    print("--- Testing all given models ---")

    prompt = generate_random_question(QUESTION_GENERATION_MODEL)
    print(f"Prompt = {prompt}")

    generations = []
    for model in models:
        print(f"\t{model.base_name} ({model.tag}) ({model.quant}) ...")
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

        generations.append(Generation(model, TEST_SYSTEM_PROMPT, prompt, response))

    save_generations(generations, parquet_output_path, also_output_xlsx=True)
    print("Models were tested & the generations were saved!\n")


if __name__ == "__main__":
    # temp testing section:
    first_2_general_models = get_general_models()[:2]
    test_models(first_2_general_models)


