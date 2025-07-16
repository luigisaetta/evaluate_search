"""
oci_models

factory to get easy access to OCI GenAI models
"""

from langchain_community.chat_models import ChatOCIGenAI
from custom_rest_embeddings import CustomRESTEmbeddings
from config import AUTH, DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE, MAX_TOKENS, ENDPOINT

from config_private import COMPARTMENT_ID

MODELS_WITHOUT_KWARGS = {
    "openai.gpt-4o-search-preview",
    "openai.gpt-4o-search-preview-2025-03-11",
}


def normalize_provider(model_id: str) -> str:
    """
    apply an hack to handle new models:
    use meta as provider for these new models
    """
    _provider = model_id.split(".")[0]

    if _provider in {"xai", "openai"}:
        # Known LangChain limitation workaround
        _provider = "meta"
    return _provider


def get_llm(
    model_id=DEFAULT_MODEL_ID, temperature=DEFAULT_TEMPERATURE, max_tokens=MAX_TOKENS
):
    """
    Initialize and return an instance of ChatOCIGenAI with the specified configuration.

    Returns:
        ChatOCIGenAI: An instance of the OCI GenAI language model.
    """
    # try to identify the provider
    _provider = normalize_provider(model_id)

    if model_id not in MODELS_WITHOUT_KWARGS:
        _model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        # for some models (OpenAI search) you cannot set those params
        _model_kwargs = None

    llm = ChatOCIGenAI(
        auth_type=AUTH,
        model_id=model_id,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=True,
        model_kwargs=_model_kwargs,
        provider=_provider,
    )
    return llm


def get_embedding_model(model_url: str, model_id: str):
    """
    Return a wrapper for the embedding model
    """
    embed_model = CustomRESTEmbeddings(model_url, model_id)

    return embed_model
