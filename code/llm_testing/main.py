import llm_tests as lt

general_models = lt.get_general_models()

lt.load_models(general_models)

lt.test_models(general_models)
