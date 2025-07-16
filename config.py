"""
All configs here
"""

DEBUG = False

# auth for OCI GenAI
AUTH = "API_KEY"

# document chunking
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

DEFAULT_MODEL_ID = "meta.llama-3.3-70b-instruct"
DEFAULT_TEMPERATURE = 0.0
MAX_TOKENS = 4000
REGION = "us-chicago-1"
ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

# LLM for generating Q&A
# QA_MODEL_ID = "openai.gpt-4.1"
QA_MODEL_ID = "xai.grok-3"

# if using NVIDIA models
EMBED_MODEL_TYPE = "NVIDIA"
# EMBED_MODEL_URL = "http://130.61.225.137:8000/v1/embeddings"
# EMBED_MODEL_ID = "nvidia/llama-3.2-nv-embedqa-1b-v2"
EMBED_MODEL_URL = "http://150.230.157.213:8000/v1/embeddings"
EMBED_MODEL_ID = "baai/bge-m3"
# EMBED_MODEL_ID = "cohere.embed-multilingual-v3.0"
# EMBED_MODEL_ID = "cohere.embed-v4.0"

# similarity search
K = 5
