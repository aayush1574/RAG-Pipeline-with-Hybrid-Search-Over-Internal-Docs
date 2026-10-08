from app.ingestion import chunk_text, deduplicate, Chunk


def test_all_chunking_strategies():
    text = "# One\n\nalpha beta\n\n# Two\n\ngamma delta"
    assert chunk_text(text, "fixed", 10, 2)
    assert chunk_text(text, "recursive", 20, 3)
    assert chunk_text(text, "semantic", 20, 0)


def test_exact_deduplication():
    chunks = [Chunk("same", "a"), Chunk("same", "b"), Chunk("different", "c")]
    assert len(deduplicate(chunks)) == 2
