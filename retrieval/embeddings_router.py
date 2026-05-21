"""
Unified embedding interface. Each provider implements .embed_documents(texts) -> List[List[float]].
Instances are lazy + cached per provider so we don't reinitialize on every call.
"""
import os
from functools import lru_cache


# Provider keys exposed to the UI. Values are the default model for each provider.
EMBEDDING_PROVIDERS = {
    "google": "models/gemini-embedding-001",
    "voyage": "voyage-3-lite",
    "openai": "text-embedding-3-small",
    "local":  "sentence-transformers/all-MiniLM-L6-v2",
}


class _GoogleAdapter:
    def __init__(self, model):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self._client = GoogleGenerativeAIEmbeddings(model=model)

    def embed_documents(self, texts):
        return self._client.embed_documents(texts)


class _VoyageAdapter:
    def __init__(self, model):
        import voyageai
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY missing in env")
        self._client = voyageai.Client()
        self._model = model

    def embed_documents(self, texts):
        # input_type="document" gets better retrieval quality on Voyage
        return self._client.embed(texts, model=self._model, input_type="document").embeddings


class _OpenAIAdapter:
    def __init__(self, model):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing in env")
        self._client = OpenAI()
        self._model = model

    def embed_documents(self, texts):
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


class _LocalAdapter:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def embed_documents(self, texts):
        return self._model.encode(texts, convert_to_numpy=True).tolist()


_ADAPTERS = {
    "google": _GoogleAdapter,
    "voyage": _VoyageAdapter,
    "openai": _OpenAIAdapter,
    "local":  _LocalAdapter,
}


@lru_cache(maxsize=None)
def get_embeddings_model(provider):
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider}. Options: {list(EMBEDDING_PROVIDERS)}")
    model = EMBEDDING_PROVIDERS[provider]
    return _ADAPTERS[provider](model)
