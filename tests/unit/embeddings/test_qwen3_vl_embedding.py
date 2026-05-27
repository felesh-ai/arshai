"""Tests for TongyiVisionEmbedding (config + input resolution).

Per Phase 2 Tasks 2.3 (config) and 2.4 (input resolution) of the forensic
video-RAG plan (`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).
"""

from unittest.mock import MagicMock

import pytest

from arshai.core.interfaces.iembedding import EmbeddingConfig
from arshai.core.interfaces.imultimodal_embedding import MultimodalInput
from arshai.embeddings.qwen3_vl_embedding import (
    TongyiVisionEmbedding,
    TongyiVisionEmbeddingConfig,
)


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


# ---------------------------------------------------------------------------
# Task 2.4 — TongyiVisionEmbedding.__init__ + _resolve_items()
# ---------------------------------------------------------------------------


def _make_client(staging_bucket="video-rag-staging", s3=None):
    cfg = TongyiVisionEmbeddingConfig(
        gateway_base_url="https://gw.example",
        gateway_token="t",
        model_name="tongyi-embedding-vision-plus",
        staging_bucket=staging_bucket,
        staging_url_ttl_seconds=3600,
    )
    s3_client = s3 or MagicMock()
    return TongyiVisionEmbedding(cfg, s3_client=s3_client), s3_client


class TestTongyiVisionEmbeddingConstructor:
    def test_stores_config_and_dependencies(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )
        s3 = MagicMock()
        c = TongyiVisionEmbedding(cfg, s3_client=s3)
        assert c.config is cfg
        assert c._s3 is s3

    def test_dimension_reflects_config(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t", dimension=768
        )
        c = TongyiVisionEmbedding(cfg, s3_client=MagicMock())
        assert c.dimension == 768


class TestResolveItems:
    @pytest.mark.asyncio
    async def test_text_only_passthrough(self):
        c, s3 = _make_client()
        items = [MultimodalInput(text="hello")]
        out = await c._resolve_items(items)
        assert len(out) == 1
        assert out[0].text == "hello"
        assert out[0].video_url is None
        assert out[0].video_path is None
        s3.upload_file.assert_not_called()
        s3.generate_presigned_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_url_passthrough(self):
        c, s3 = _make_client()
        items = [MultimodalInput(image_url="https://x/y.jpg")]
        out = await c._resolve_items(items)
        assert out[0].image_url == "https://x/y.jpg"
        s3.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_video_url_passthrough(self):
        c, s3 = _make_client()
        items = [MultimodalInput(video_url="https://x/clip.mp4")]
        out = await c._resolve_items(items)
        assert out[0].video_url == "https://x/clip.mp4"
        assert out[0].video_path is None
        s3.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_video_path_uploads_and_signs(self):
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://staging.example/signed?sig=abc"
        c, _ = _make_client(s3=s3)

        items = [MultimodalInput(video_path="/tmp/clip.mp4")]
        out = await c._resolve_items(items)

        assert s3.upload_file.call_count == 1
        bucket = c.config.staging_bucket
        # Positional or kwarg call — accept both.
        args, kwargs = s3.upload_file.call_args
        assert "/tmp/clip.mp4" in (args[0:1] or (kwargs.get("Filename"),))
        assert bucket in (args[1:2] or (kwargs.get("Bucket"),))

        s3.generate_presigned_url.assert_called_once()
        gpu_args, gpu_kwargs = s3.generate_presigned_url.call_args
        assert gpu_args[0] == "get_object" or gpu_kwargs.get("ClientMethod") == "get_object"
        assert gpu_kwargs.get("ExpiresIn") == c.config.staging_url_ttl_seconds

        # The resolved item swaps video_path → video_url.
        assert out[0].video_url == "https://staging.example/signed?sig=abc"
        assert out[0].video_path is None

    @pytest.mark.asyncio
    async def test_video_path_without_staging_bucket_raises(self):
        cfg = TongyiVisionEmbeddingConfig(
            gateway_base_url="https://x", gateway_token="t"
        )  # staging_bucket=None
        c = TongyiVisionEmbedding(cfg, s3_client=MagicMock())
        with pytest.raises(ValueError, match="staging_bucket"):
            await c._resolve_items([MultimodalInput(video_path="/tmp/c.mp4")])

    @pytest.mark.asyncio
    async def test_mixed_batch_resolves_only_video_paths(self):
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://signed/url"
        c, _ = _make_client(s3=s3)
        items = [
            MultimodalInput(text="caption"),
            MultimodalInput(image_url="https://x/y.jpg"),
            MultimodalInput(video_path="/tmp/clip.mp4"),
        ]
        out = await c._resolve_items(items)
        assert len(out) == 3
        assert out[0].text == "caption"
        assert out[1].image_url == "https://x/y.jpg"
        assert out[2].video_url == "https://signed/url"
        # Only one upload — for the video_path item.
        assert s3.upload_file.call_count == 1
