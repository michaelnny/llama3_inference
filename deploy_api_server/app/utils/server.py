"""Maps user friendly model names to Triton model names"""

TRITON_MODEL_NAME_MAP = {
    "llama3": "llama3_ensemble",
    "llama3-7b": "llama3_ensemble",
    "text-embedding": "st_ensemble",
    "embedding": "st_ensemble",
}


DEFAULT_LLM_MODEL_NAME = "llama3_ensemble"
DEFAULT_EMBED_MODEL_NAME = "st_ensemble"


def get_triton_server_model_name(model_name: str, for_embedding: bool = False) -> str:
    # Map model name to triton server model name
    if model_name is None or model_name not in TRITON_MODEL_NAME_MAP:
        if for_embedding:
            return DEFAULT_EMBED_MODEL_NAME
        else:
            return DEFAULT_LLM_MODEL_NAME
    else:
        return TRITON_MODEL_NAME_MAP[model_name]
