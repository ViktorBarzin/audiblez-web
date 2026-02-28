"""Voice cloning API routes.

Endpoints for managing cloned voices, YouTube audio search/download,
recording uploads, and audio transcription.
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from api.auth import User, get_current_user
from models.voice_schemas import (
    ClonedVoiceCreate,
    TranscriptionResult,
    YouTubeDownloadRequest,
    YouTubeSearchResult,
)
from services import voice_cloner
from services.qwen_tts import generate_sample, get_available_models, iso_to_language
from services.transcriber import transcribe
from services.youtube_search import download_audio, search_youtube

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/mnt/temp"))


def _user_temp_dir(user_id: str) -> Path:
    """Return (and create) the per-user temp directory."""
    d = TEMP_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_most_recent_audio(temp_dir: Path) -> Path | None:
    """Find the most recent ref_*.wav or recording_*.wav file in *temp_dir*."""
    candidates = list(temp_dir.glob("ref_*.wav")) + list(temp_dir.glob("recording_*.wav"))
    # Also include YouTube-downloaded WAV files
    candidates += [
        f for f in temp_dir.glob("*.wav")
        if not f.name.startswith("ref_") and not f.name.startswith("recording_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _cleanup_temp(temp_dir: Path) -> None:
    """Remove all files from a user's temp directory."""
    if temp_dir.exists():
        for f in temp_dir.iterdir():
            if f.is_file():
                f.unlink()
        logger.info("Cleaned up temp directory %s", temp_dir)


# ============================================================================
# Cloned Voice CRUD
# ============================================================================


@router.get("/cloned-voices")
async def list_cloned_voices(user: User = Depends(get_current_user)):
    """List all public voices and the current user's cloned voices."""
    voices = await voice_cloner.list_voices(user.uid)
    return voices


@router.get("/cloned-voices/models")
async def list_models():
    """List available Qwen3-TTS models. No auth required."""
    return get_available_models()


@router.get("/cloned-voices/{voice_id}")
async def get_cloned_voice(voice_id: str, user: User = Depends(get_current_user)):
    """Get a single cloned voice by ID."""
    voice = await voice_cloner.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    return voice


@router.post("/cloned-voices")
async def create_cloned_voice(
    body: ClonedVoiceCreate,
    user: User = Depends(get_current_user),
):
    """Create a new cloned voice from uploaded/recorded/downloaded audio.

    Looks for the most recent audio file in the user's temp directory
    (uploaded via recording/upload or youtube/download), creates the
    voice clone, and cleans up temp files afterwards.
    """
    temp_dir = _user_temp_dir(user.uid)
    audio_file = _find_most_recent_audio(temp_dir)

    if not audio_file:
        raise HTTPException(
            status_code=400,
            detail="No audio file found. Upload a recording or download from YouTube first.",
        )

    try:
        voice = await voice_cloner.create_voice(
            name=body.name,
            user_id=user.uid,
            source_type=body.source_type,
            language=body.language,
            ref_audio_path=audio_file,
            transcript=body.transcript,
            model_id=body.model_id,
            source_url=body.source_url,
        )
    finally:
        _cleanup_temp(temp_dir)

    return voice


@router.delete("/cloned-voices/{voice_id}")
async def delete_cloned_voice(voice_id: str, user: User = Depends(get_current_user)):
    """Delete a cloned voice. Only the creator may delete it."""
    deleted = await voice_cloner.delete_voice(voice_id, user.uid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found or not owned by you")
    return {"status": "deleted"}


@router.get("/cloned-voices/{voice_id}/sample")
async def get_cloned_voice_sample(
    voice_id: str,
    user: User = Depends(get_current_user),
):
    """Generate a TTS preview sample for a cloned voice."""
    voice = await voice_cloner.get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    clone_prompt_bytes = voice_cloner.get_clone_prompt_bytes(voice_id)
    if not clone_prompt_bytes:
        raise HTTPException(status_code=404, detail="Clone prompt not found for this voice")

    language = iso_to_language(voice.language)
    wav_bytes = generate_sample(
        clone_prompt_bytes=clone_prompt_bytes,
        language=language,
    )

    return Response(content=wav_bytes, media_type="audio/wav")


# ============================================================================
# YouTube Search & Download
# ============================================================================


@router.get("/youtube/search", response_model=list[YouTubeSearchResult])
async def youtube_search(
    q: str = Query(..., min_length=1, max_length=200),
    user: User = Depends(get_current_user),
):
    """Search YouTube for videos matching the query.

    Appends ' interview' to the query to bias results towards spoken content
    suitable for voice cloning references.
    """
    search_query = f"{q} interview"
    results = await search_youtube(search_query)
    return [
        YouTubeSearchResult(
            video_id=r.video_id,
            title=r.title,
            duration_seconds=r.duration_seconds,
            url=r.url,
            thumbnail=r.thumbnail,
        )
        for r in results
    ]


@router.post("/youtube/download")
async def youtube_download(
    body: YouTubeDownloadRequest,
    user: User = Depends(get_current_user),
):
    """Download audio from a YouTube video to the user's temp directory."""
    temp_dir = _user_temp_dir(user.uid)

    try:
        output_path = await download_audio(
            video_url=body.video_url,
            output_dir=temp_dir,
            max_duration_seconds=body.max_duration_seconds,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Rename to ref_*.wav so _find_most_recent_audio picks it up
    ref_path = temp_dir / f"ref_{uuid.uuid4().hex[:8]}.wav"
    shutil.move(str(output_path), str(ref_path))

    return {"status": "downloaded", "filename": ref_path.name}


# ============================================================================
# Recording & Transcription
# ============================================================================


@router.post("/recording/upload")
async def upload_recording(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload an audio recording to the user's temp directory."""
    temp_dir = _user_temp_dir(user.uid)
    filename = f"recording_{uuid.uuid4().hex[:8]}.wav"
    dest = temp_dir / filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "uploaded", "filename": filename}


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(user: User = Depends(get_current_user)):
    """Transcribe the most recent audio file in the user's temp directory."""
    temp_dir = _user_temp_dir(user.uid)
    audio_file = _find_most_recent_audio(temp_dir)

    if not audio_file:
        raise HTTPException(
            status_code=400,
            detail="No audio file found. Upload a recording or download from YouTube first.",
        )

    result = transcribe(audio_file)

    return TranscriptionResult(
        text=result["text"],
        language=result["language"],
        segments=result.get("segments", []),
    )
