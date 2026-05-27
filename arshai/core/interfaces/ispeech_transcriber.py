"""Speech-to-text transcription interface.

Added in Phase 2 Task 2.2 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).

The v1 implementation behind this protocol is ``Qwen3ASR``, which calls
the AI Gateway's ``POST /v1/audio/transcriptions`` route (URL-driven,
JSON body — see AI-Gateway ADR 0007). Other providers (e.g. local
faster-whisper) may follow.

The contract is intentionally URL-only on the input side: implementations
do not accept inline audio bytes. Callers that have bytes in hand stage
them to object storage and pass a signed URL. This keeps the gateway —
and the framework — out of the audio-staging business; see ADR 0007 for
the rationale.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ISpeechTranscriber(Protocol):
    """Interface for speech-to-text services.

    Return contract
    ---------------
    ``transcribe()`` returns a ``Dict[str, Any]`` with at minimum::

        {
            "text": str,                 # transcript
            "language": Optional[str],   # detected or echoed language code
            "segments": List[Dict],      # may be empty if upstream doesn't segment
            "upstream_request_id": str,  # for forensic correlation
        }

    The ``segments`` list mirrors OpenAI's ``verbose_json`` shape but may be
    empty for providers that return only a single transcript chunk (notably
    DashScope's ``qwen3-asr-flash``).
    """

    async def transcribe(
        self,
        audio: str,
        language: Optional[str] = None,
        response_format: str = "json",
        hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio: Externally-fetchable audio URL. Implementations do NOT
                accept inline bytes — callers stage to object storage first.
            language: ISO 639-1 hint (e.g. ``"en"``, ``"fa"``). When omitted
                the upstream is asked to auto-detect; auto-detect is
                unreliable on short clips, so callers should pass a hint
                when they know it.
            response_format: ``"json"`` (default) or ``"verbose_json"``. Note
                that ``segments`` and ``words`` are not populated by all
                providers regardless of format — see provider docs.
            hints: Optional vocabulary biasing list (proper nouns, jargon).
                Providers that don't support biasing silently ignore.

        Returns:
            See class docstring for the return contract.
        """
        ...
