from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    data_dir: Path = Path("./data")
    chunk_strategy: str = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 12
    rerank_top_k: int = 5
    semantic_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    dedup_threshold: float = 0.95
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
