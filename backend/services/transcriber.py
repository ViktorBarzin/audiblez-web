"""Whisper-based transcription service for voice reference audio clips.

Transcribes short audio clips (3-30s) so the transcript can be paired
with the audio to build a Qwen3-TTS voice-clone prompt.
"""

from __future__ import annotations

import os
from pathlib import Path

import whisper

WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "small")

_model = None


def _get_model():
    """Load the Whisper model lazily and keep it in memory."""
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe(audio_path: Path, language: str | None = None) -> dict:
    """Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        language: Optional language code (e.g. "en", "zh") to constrain
                  language detection.

    Returns:
        Dict with keys:
            text     - full transcript string
            language - detected (or forced) language code
            segments - list of ``{start, end, text}`` dicts
    """
    model = _get_model()

    options: dict = {}
    if language is not None:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ],
    }
