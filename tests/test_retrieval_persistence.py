from app.ingestion import Chunk
from app.retrieval import HybridRetriever


def test_index_round_trip(tmp_path):
    path = tmp_path / "index.json"
    original = HybridRetriever()
    original.add([Chunk("access policy", "handbook.md", 2, "Access")], [[1.0, 0.0]])
    original.save(path)

    restored = HybridRetriever()
    restored.load(path)
    assert restored.chunks[0].source == "handbook.md"
    assert restored.embeddings.tolist() == [[1.0, 0.0]]
    assert restored.search("access", [1.0, 0.0])[0]["source"] == "handbook.md"
