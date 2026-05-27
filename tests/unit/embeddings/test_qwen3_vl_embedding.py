"""Tests for TongyiVisionEmbeddingConfig.

Per Phase 2 Task 2.3 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).
"""

import pytest

from arshai.core.interfaces.iembedding import EmbeddingConfig
from arshai.embeddings.qwen3_vl_embedding import TongyiVisionEmbeddingConfig


class TestTongyiVisionEmbeddingConfigDefaults:
    def test_minimal_construction_with_explicit_credentials(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://gateway.example.com",
            gateway_token="t",
        )
        assert cfg.gateway_base_url == "https://gateway.example.com"
        assert cfg.gateway_token == "t"

    def test_default_dimension_is_1024(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        assert cfg.dimension == 1024

    def test_default_batch_limit_is_10(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        assert cfg.batch_limit == 10

    def test_default_max_concurrency_is_4(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        assert cfg.max_concurrency == 4

    def test_default_staging_url_ttl_seconds_is_86400(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        assert cfg.staging_url_ttl_seconds == 86400

    def test_staging_bucket_default_is_none(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        assert cfg.staging_bucket is None

    def test_explicit_overrides_take_effect(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x",
            gateway_token="t",
            dimension=512,
            batch_limit=4,
            max_concurrency=8,
            staging_bucket="video-rag-staging",
            staging_url_ttl_seconds=3600,
        )
        assert cfg.dimension == 512
        assert cfg.batch_limit == 4
        assert cfg.max_concurrency == 8
        assert cfg.staging_bucket == "video-rag-staging"
        assert cfg.staging_url_ttl_seconds == 3600


class TestEnvFallback:
    def test_gateway_base_url_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_BASE_URL", "https://from-env.example")
        monkeypatch.setenv("GATEWAY_TOKEN", "env-token")
        cfg = TongyiVisionEmbeddingConfig()
        assert cfg.gateway_base_url == "https://from-env.example"

    def test_gateway_token_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_BASE_URL", "https://x")
        monkeypatch.setenv("GATEWAY_TOKEN", "env-token")
        cfg = TongyiVisionEmbeddingConfig()
        assert cfg.gateway_token == "env-token"

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_BASE_URL", "https://env")
        monkeypatch.setenv("GATEWAY_TOKEN", "env-tok")
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://explicit", gateway_token="explicit-tok"
        )
        assert cfg.gateway_base_url == "https://explicit"
        assert cfg.gateway_token == "explicit-tok"

    def test_no_explicit_no_env_leaves_credentials_none(self, monkeypatch):
        # Mirrors CloudflareGatewayEmbeddingConfig: the missing-credential
        # check is enforced by the implementation at runtime, not by the
        # config DTO at construction.
        monkeypatch.delenv("GATEWAY_BASE_URL", raising=False)
        monkeypatch.delenv("GATEWAY_TOKEN", raising=False)
        cfg = TongyiVisionEmbeddingConfig()
        assert cfg.gateway_base_url is None
        assert cfg.gateway_token is None


class TestInheritance:
    def test_extends_embedding_config(self):
        assert issubclass(TongyiVisionEmbeddingConfig, EmbeddingConfig)

    def test_inherits_embedding_config_fields(self):
        # EmbeddingConfig contributes model_name / batch_size / additional_params.
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x",
            gateway_token="t",
            model_name="tongyi-embedding-vision-plus",
        )
        assert cfg.model_name == "tongyi-embedding-vision-plus"
