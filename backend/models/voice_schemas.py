"""Pydantic models for the voice cloning API."""

from pydantic import BaseModel, Field
from typing import Optional


class ClonedVoice(BaseModel):
    """A cloned voice stored in the database."""

    id: str
    name: str
    created_by: str
    source_type: str
    source_url: Optional[str] = None
    language: str
    ref_audio: str
    transcript: str
    model_id: str
    is_public: bool = True
    created_at: Optional[str] = None


class ClonedVoiceCreate(BaseModel):
    """Request body for creating a new cloned voice."""

    name: str = Field(min_length=1, max_length=100)
    language: str = Field(min_length=2, max_length=5)
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    transcript: str = Field(min_length=1)
    source_type: str = Field(pattern="^(youtube|upload|recording)$")
    source_url: Optional[str] = None


class YouTubeSearchResult(BaseModel):
    """A single YouTube video search result."""

    video_id: str
    title: str
    duration_seconds: int
    url: str
    thumbnail: str


class YouTubeDownloadRequest(BaseModel):
    """Request body for downloading audio from a YouTube video."""

    video_url: str
    max_duration_seconds: int = Field(default=30, ge=3, le=60)


class TranscriptionResult(BaseModel):
    """Result of transcribing an audio file via Whisper."""

    text: str
    language: str
    segments: list[dict] = []
