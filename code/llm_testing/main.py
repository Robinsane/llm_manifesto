import llm_tests as lt

general_models = lt.get_general_models()

# Loading all models in RAM upfront once <-> 1TB RAM + OLLAMA_KEEP_ALIVE=-1
lt.load_models(general_models)

while True:
    lt.test_models(general_models)
