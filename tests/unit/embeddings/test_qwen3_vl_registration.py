"""Smoke test for Task 2.8 — TongyiVisionEmbedding is exported from
``arshai.embeddings`` at the package level."""


def test_tongyi_vision_embedding_importable_at_package_level():
    from arshai.embeddings import TongyiVisionEmbedding, TongyiVisionEmbeddingConfig

    assert TongyiVisionEmbedding is not None
    assert TongyiVisionEmbeddingConfig is not None


def test_listed_in_package_all():
    import arshai.embeddings as pkg

    assert "TongyiVisionEmbedding" in pkg.__all__
    assert "TongyiVisionEmbeddingConfig" in pkg.__all__
