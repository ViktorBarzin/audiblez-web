# Voice Cloning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add voice cloning to audiblez-web using Qwen3-TTS, with YouTube search, file upload, and mic recording as voice sources.

**Architecture:** Dual pipeline — existing audiblez CLI for Kokoro preset voices (untouched), new Python-native pipeline for Qwen3-TTS cloned voices. SQLite for voice metadata, NFS for audio files. All new dependencies embedded in the existing container.

**Tech Stack:** Python 3.11+ (FastAPI, qwen-tts, openai-whisper, yt-dlp, aiosqlite), Svelte 5, SQLite, FFmpeg

**Design doc:** `docs/plans/2026-02-28-voice-cloning-design.md`

---

### Task 1: SQLite Database Layer

**Files:**
- Create: `backend/services/database.py`
- Modify: `backend/main.py:1-41`
- Modify: `backend/requirements.txt`

**Step 1: Add aiosqlite dependency**

Add to `backend/requirements.txt`:
```
aiosqlite>=0.20.0
```

**Step 2: Create database service**

Create `backend/services/database.py`:

```python
"""SQLite database for cloned voice profiles.

Uses WAL mode for concurrent read access and an asyncio lock
for serializing writes. The database file lives on NFS at the
path specified by VOICES_DB env var.
"""

import asyncio
import os
import sqlite3
from pathlib import Path

VOICES_DB = os.environ.get("VOICES_DB", "/mnt/voices/voices.db")

_write_lock = asyncio.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_by  TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('youtube', 'upload', 'recording', 'curated')),
    source_url  TEXT,
    language    TEXT NOT NULL,
    ref_audio   TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    is_public   BOOLEAN DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection with WAL mode."""
    db_path = Path(VOICES_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist. Called on app startup."""
    conn = _get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


async def execute_read(query: str, params: tuple = ()) -> list[dict]:
    """Execute a read query and return results as list of dicts."""
    conn = _get_connection()
    try:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


async def execute_write(query: str, params: tuple = ()) -> None:
    """Execute a write query with the global write lock."""
    async with _write_lock:
        conn = _get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
        finally:
            conn.close()


async def execute_write_returning(query: str, params: tuple = ()) -> dict | None:
    """Execute a write query and return the affected row."""
    async with _write_lock:
        conn = _get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            # For INSERT, fetch the row back
            cursor = conn.execute(
                "SELECT * FROM voices WHERE id = ?", (params[0],)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
```

**Step 3: Initialize DB on app startup**

Modify `backend/main.py` — add after the FastAPI app creation (after line 9):

```python
from services.database import init_db

@app.on_event("startup")
async def startup():
    init_db()
```

**Step 4: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/database.py backend/main.py backend/requirements.txt
git commit -m "$(cat <<'EOF'
feat: add SQLite database layer for cloned voice profiles

WAL mode for concurrent reads, asyncio lock for write serialization.
DB file stored on NFS at VOICES_DB env var path.
EOF
)"
```

---

### Task 2: YouTube Search Service

**Files:**
- Create: `backend/services/youtube_search.py`
- Modify: `backend/requirements.txt`

**Step 1: Add yt-dlp dependency**

Add to `backend/requirements.txt`:
```
yt-dlp>=2024.0.0
```

**Step 2: Create YouTube search service**

Create `backend/services/youtube_search.py`:

```python
"""YouTube search and audio download service using yt-dlp.

Searches YouTube for voice samples and downloads/trims audio
for use as Qwen3-TTS voice cloning references.
"""

import asyncio
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment


@dataclass
class YouTubeResult:
    video_id: str
    title: str
    duration_seconds: int
    url: str
    thumbnail: str


async def search_youtube(query: str, max_results: int = 5) -> list[YouTubeResult]:
    """Search YouTube and return video metadata without downloading.

    Uses yt-dlp's search feature: ytsearch{N}:{query}
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{max_results}:{query}",
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"yt-dlp search failed: {stderr.decode()}")

    results = []
    for line in stdout.decode().strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        results.append(YouTubeResult(
            video_id=data.get("id", ""),
            title=data.get("title", "Unknown"),
            duration_seconds=int(data.get("duration", 0) or 0),
            url=f"https://www.youtube.com/watch?v={data.get('id', '')}",
            thumbnail=data.get("thumbnail", ""),
        ))

    return results


async def download_audio(
    video_url: str,
    output_dir: Path,
    max_duration_seconds: int = 30,
) -> Path:
    """Download audio from a YouTube video and trim to max_duration_seconds.

    Returns path to the trimmed WAV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / f"yt_{uuid.uuid4().hex}.wav"

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(temp_path.with_suffix(".%(ext)s")),
        "--no-playlist",
        video_url,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {stderr.decode()}")

    # Find the downloaded file (yt-dlp may change extension)
    downloaded = None
    for f in output_dir.glob(f"yt_{temp_path.stem.split('_', 1)[1]}*"):
        downloaded = f
        break

    if not downloaded or not downloaded.exists():
        # Try the exact path
        downloaded = temp_path
        if not downloaded.exists():
            raise FileNotFoundError("Downloaded audio file not found")

    # Trim to max_duration_seconds using pydub
    audio = AudioSegment.from_file(str(downloaded))
    trimmed = audio[: max_duration_seconds * 1000]

    final_path = output_dir / f"ref_{uuid.uuid4().hex}.wav"
    trimmed.export(str(final_path), format="wav")

    # Clean up original if different
    if downloaded != final_path and downloaded.exists():
        downloaded.unlink()

    return final_path
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/youtube_search.py backend/requirements.txt
git commit -m "$(cat <<'EOF'
feat: add YouTube search and audio download service

Uses yt-dlp for searching and downloading voice reference samples.
Trims downloads to 30 seconds for voice cloning input.
EOF
)"
```

---

### Task 3: Whisper Transcription Service

**Files:**
- Create: `backend/services/transcriber.py`
- Modify: `backend/requirements.txt`

**Step 1: Add whisper dependency**

Add to `backend/requirements.txt`:
```
openai-whisper>=20231117
```

**Step 2: Create transcription service**

Create `backend/services/transcriber.py`:

```python
"""Audio transcription service using OpenAI Whisper.

Loads the Whisper model lazily on first use and keeps it in memory.
Model size is configurable via WHISPER_MODEL env var.
"""

import os
from pathlib import Path

import whisper

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

_model = None


def _get_model():
    """Load Whisper model lazily, keep in memory."""
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe(audio_path: Path, language: str | None = None) -> dict:
    """Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language: Optional ISO 639-1 language code to constrain detection.

    Returns:
        dict with keys: text (full transcript), language (detected), segments
    """
    model = _get_model()

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "en"),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
    }
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/transcriber.py backend/requirements.txt
git commit -m "$(cat <<'EOF'
feat: add Whisper transcription service

Lazy-loads Whisper model on first use. Model size configurable via
WHISPER_MODEL env var (default: small, ~1GB VRAM).
EOF
)"
```

---

### Task 4: Qwen3-TTS Service

**Files:**
- Create: `backend/services/qwen_tts.py`
- Modify: `backend/requirements.txt`

**Step 1: Add qwen-tts dependency**

Add to `backend/requirements.txt`:
```
qwen-tts>=0.1.0
```

**Step 2: Create Qwen3-TTS service**

Create `backend/services/qwen_tts.py`:

```python
"""Qwen3-TTS model management, voice cloning, and speech generation.

Loads the model lazily on first use and keeps it in GPU memory.
Model ID is configurable via QWEN_TTS_MODEL env var.

Key operations:
- create_clone_prompt: Create a reusable voice clone prompt from reference audio
- generate_speech: Generate speech from text using a cloned voice
- generate_sample: Generate a short preview sample for a cloned voice
"""

import io
import os
import pickle
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

QWEN_TTS_MODEL = os.environ.get(
    "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

# Available models with VRAM estimates
AVAILABLE_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {"name": "1.7B Base", "vram_gb": 5},
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {"name": "0.6B Base", "vram_gb": 2},
}

_model = None
_loaded_model_id = None


def _get_model(model_id: str | None = None):
    """Load Qwen3-TTS model lazily, keep in GPU memory."""
    global _model, _loaded_model_id

    target_model = model_id or QWEN_TTS_MODEL

    if _model is not None and _loaded_model_id == target_model:
        return _model

    # Unload previous model if switching
    if _model is not None:
        del _model
        torch.cuda.empty_cache()

    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _model = Qwen3TTSModel.from_pretrained(
        target_model,
        device_map=device,
        dtype=dtype,
    )
    _loaded_model_id = target_model
    return _model


def create_clone_prompt(
    ref_audio_path: Path,
    ref_text: str,
    model_id: str | None = None,
) -> bytes:
    """Create a reusable voice clone prompt from reference audio + transcript.

    Args:
        ref_audio_path: Path to reference WAV file (3-30 seconds)
        ref_text: Transcript of the reference audio
        model_id: Optional model ID override

    Returns:
        Serialized clone prompt (bytes) for storage
    """
    model = _get_model(model_id)
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )
    return pickle.dumps(prompt)


def generate_speech(
    text: str,
    clone_prompt_bytes: bytes,
    language: str = "English",
    model_id: str | None = None,
) -> tuple[np.ndarray, int]:
    """Generate speech from text using a cloned voice.

    Args:
        text: Text to synthesize
        clone_prompt_bytes: Serialized clone prompt from create_clone_prompt()
        language: Language name (e.g., "English", "Chinese")
        model_id: Optional model ID override

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    model = _get_model(model_id)
    prompt = pickle.loads(clone_prompt_bytes)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt,
    )

    return wavs[0], sr


def generate_chapter_audio(
    chapter_text: str,
    clone_prompt_bytes: bytes,
    output_path: Path,
    language: str = "English",
    model_id: str | None = None,
    speed: float = 1.0,
    max_segment_words: int = 500,
) -> Path:
    """Generate audio for a full chapter, splitting long text into segments.

    Args:
        chapter_text: Full chapter text
        clone_prompt_bytes: Serialized clone prompt
        output_path: Where to save the output WAV
        language: Language name
        model_id: Optional model ID override
        speed: Playback speed multiplier (applied via resampling)
        max_segment_words: Maximum words per TTS segment

    Returns:
        Path to the generated WAV file
    """
    segments = _split_text(chapter_text, max_segment_words)
    all_audio = []
    sample_rate = None

    for segment in segments:
        if not segment.strip():
            continue

        audio, sr = generate_speech(
            text=segment,
            clone_prompt_bytes=clone_prompt_bytes,
            language=language,
            model_id=model_id,
        )
        all_audio.append(audio)
        sample_rate = sr

    if not all_audio or sample_rate is None:
        raise RuntimeError("No audio generated for chapter")

    # Concatenate all segments
    combined = np.concatenate(all_audio)

    # Apply speed adjustment via resampling
    if speed != 1.0 and speed > 0:
        # Write to buffer, adjust with pydub, read back
        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV")
        buf.seek(0)
        audio_seg = AudioSegment.from_file(buf, format="wav")

        # Change speed by altering frame rate then converting back
        adjusted = audio_seg._spawn(
            audio_seg.raw_data,
            overrides={"frame_rate": int(audio_seg.frame_rate * speed)},
        )
        adjusted = adjusted.set_frame_rate(sample_rate)
        adjusted.export(str(output_path), format="wav")
    else:
        sf.write(str(output_path), combined, sample_rate)

    return output_path


def generate_sample(
    clone_prompt_bytes: bytes,
    language: str = "English",
    model_id: str | None = None,
) -> bytes:
    """Generate a short preview sample for a cloned voice.

    Returns WAV bytes.
    """
    sample_texts = {
        "English": "The quick brown fox jumps over the lazy dog. A wonderful evening awaits us all.",
        "Chinese": "快速的棕色狐狸跳过了懒惰的狗。美好的夜晚等待着我们。",
        "Japanese": "素早い茶色の狐が怠惰な犬を飛び越えます。素晴らしい夜が待っています。",
        "Korean": "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "German": "Der schnelle braune Fuchs springt uber den faulen Hund.",
        "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Russian": "Быстрая коричневая лисица перепрыгивает через ленивую собаку.",
        "Portuguese": "A rapida raposa marrom pula sobre o cachorro preguicoso.",
        "Spanish": "El rapido zorro marron salta sobre el perro perezoso.",
        "Italian": "La veloce volpe marrone salta sopra il cane pigro.",
    }

    text = sample_texts.get(language, sample_texts["English"])
    audio, sr = generate_speech(text, clone_prompt_bytes, language, model_id)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def get_available_models() -> list[dict]:
    """Return list of available models with VRAM estimates."""
    return [
        {"id": model_id, **info}
        for model_id, info in AVAILABLE_MODELS.items()
    ]


# Language name mapping (ISO 639-1 -> Qwen3-TTS language name)
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def iso_to_language(iso_code: str) -> str:
    """Convert ISO 639-1 code to Qwen3-TTS language name."""
    return LANGUAGE_MAP.get(iso_code, "English")


def _split_text(text: str, max_words: int = 500) -> list[str]:
    """Split text into segments at sentence boundaries.

    Tries to keep segments under max_words while respecting
    sentence boundaries (periods, question marks, exclamation marks).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = [sentence]
            current_word_count = word_count
        else:
            current_segment.append(sentence)
            current_word_count += word_count

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/qwen_tts.py backend/requirements.txt
git commit -m "$(cat <<'EOF'
feat: add Qwen3-TTS service for voice cloning and speech generation

Lazy model loading, clone prompt creation/serialization, chapter-level
generation with text splitting, speed adjustment via resampling.
Configurable model via QWEN_TTS_MODEL env var.
EOF
)"
```

---

### Task 5: Voice Cloner Service (CRUD + Orchestration)

**Files:**
- Create: `backend/services/voice_cloner.py`
- Create: `backend/models/voice_schemas.py`

**Step 1: Create voice Pydantic models**

Create `backend/models/voice_schemas.py`:

```python
"""Pydantic models for cloned voice API."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ClonedVoice(BaseModel):
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
    name: str = Field(min_length=1, max_length=100)
    language: str = Field(min_length=2, max_length=5)
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    transcript: str = Field(min_length=1)
    source_type: str = Field(pattern="^(youtube|upload|recording)$")
    source_url: Optional[str] = None


class YouTubeSearchResult(BaseModel):
    video_id: str
    title: str
    duration_seconds: int
    url: str
    thumbnail: str


class YouTubeDownloadRequest(BaseModel):
    video_url: str
    max_duration_seconds: int = Field(default=30, ge=3, le=60)


class TranscriptionResult(BaseModel):
    text: str
    language: str
    segments: list[dict] = []
```

**Step 2: Create voice cloner service**

Create `backend/services/voice_cloner.py`:

```python
"""Voice cloner service — CRUD operations and clone workflow orchestration.

Coordinates between database, NFS storage, Qwen3-TTS, and Whisper
to create, store, and manage cloned voice profiles.
"""

import shutil
import uuid
from pathlib import Path

from services.database import execute_read, execute_write, execute_write_returning
from services.qwen_tts import create_clone_prompt, iso_to_language
from models.voice_schemas import ClonedVoice

VOICES_DIR = Path("/mnt/voices")


def _voice_dir(voice_id: str) -> Path:
    """Get the NFS directory for a voice's files."""
    d = VOICES_DIR / voice_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def list_voices(user_id: str) -> list[ClonedVoice]:
    """List all public voices plus the user's own private voices."""
    rows = await execute_read(
        "SELECT * FROM voices WHERE is_public = 1 OR created_by = ? ORDER BY created_at DESC",
        (user_id,),
    )
    return [ClonedVoice(**row) for row in rows]


async def get_voice(voice_id: str) -> ClonedVoice | None:
    """Get a single voice by ID."""
    rows = await execute_read("SELECT * FROM voices WHERE id = ?", (voice_id,))
    if rows:
        return ClonedVoice(**rows[0])
    return None


async def create_voice(
    name: str,
    user_id: str,
    source_type: str,
    language: str,
    ref_audio_path: Path,
    transcript: str,
    model_id: str,
    source_url: str | None = None,
) -> ClonedVoice:
    """Create a new cloned voice profile.

    1. Copy reference audio to NFS
    2. Generate clone prompt via Qwen3-TTS
    3. Save clone prompt to NFS
    4. Insert record into SQLite
    """
    voice_id = str(uuid.uuid4())
    voice_dir = _voice_dir(voice_id)

    # Copy reference audio to voice directory
    ref_dest = voice_dir / "ref.wav"
    shutil.copy2(str(ref_audio_path), str(ref_dest))

    # Generate and save clone prompt
    lang_name = iso_to_language(language)
    prompt_bytes = create_clone_prompt(ref_dest, transcript, model_id)
    prompt_path = voice_dir / "clone_prompt.bin"
    prompt_path.write_bytes(prompt_bytes)

    # Insert into database
    await execute_write(
        """INSERT INTO voices (id, name, created_by, source_type, source_url,
           language, ref_audio, transcript, model_id, is_public)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (voice_id, name, user_id, source_type, source_url,
         language, str(ref_dest), transcript, model_id),
    )

    return await get_voice(voice_id)


async def delete_voice(voice_id: str, user_id: str) -> bool:
    """Delete a voice. Only the creator can delete."""
    voice = await get_voice(voice_id)
    if not voice or voice.created_by != user_id:
        return False

    # Delete NFS files
    voice_dir = _voice_dir(voice_id)
    if voice_dir.exists():
        shutil.rmtree(voice_dir)

    # Delete from database
    await execute_write("DELETE FROM voices WHERE id = ?", (voice_id,))
    return True


def get_clone_prompt_bytes(voice_id: str) -> bytes | None:
    """Load the serialized clone prompt from NFS."""
    prompt_path = _voice_dir(voice_id) / "clone_prompt.bin"
    if prompt_path.exists():
        return prompt_path.read_bytes()
    return None


def is_cloned_voice(voice_id: str) -> bool:
    """Check if a voice ID looks like a cloned voice UUID (vs a Kokoro voice ID)."""
    try:
        uuid.UUID(voice_id)
        return True
    except ValueError:
        return False
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/voice_cloner.py backend/models/voice_schemas.py
git commit -m "$(cat <<'EOF'
feat: add voice cloner service and voice schemas

CRUD operations for cloned voices, NFS storage management,
clone prompt generation orchestration. Includes Pydantic models
for the voice cloning API.
EOF
)"
```

---

### Task 6: Voice API Routes

**Files:**
- Create: `backend/api/voice_routes.py`
- Modify: `backend/main.py`

**Step 1: Create voice routes**

Create `backend/api/voice_routes.py`:

```python
"""API routes for cloned voice management, YouTube search, and transcription."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from api.auth import User, get_current_user
from models.voice_schemas import (
    ClonedVoice,
    ClonedVoiceCreate,
    TranscriptionResult,
    YouTubeDownloadRequest,
    YouTubeSearchResult,
)
from services.voice_cloner import (
    create_voice,
    delete_voice,
    get_clone_prompt_bytes,
    get_voice,
    list_voices,
)
from services.qwen_tts import generate_sample, get_available_models, iso_to_language
from services.youtube_search import download_audio, search_youtube
from services.transcriber import transcribe

router = APIRouter(prefix="/api")


# ============================================================================
# Cloned Voice CRUD
# ============================================================================

@router.get("/cloned-voices", response_model=list[ClonedVoice])
async def list_cloned_voices(user: User = Depends(get_current_user)):
    """List all public cloned voices plus user's own."""
    return await list_voices(user.uid)


@router.get("/cloned-voices/models")
async def list_models():
    """List available Qwen3-TTS models with VRAM estimates."""
    return get_available_models()


@router.get("/cloned-voices/{voice_id}", response_model=ClonedVoice)
async def get_cloned_voice(voice_id: str, user: User = Depends(get_current_user)):
    """Get a specific cloned voice."""
    voice = await get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    return voice


@router.post("/cloned-voices", response_model=ClonedVoice)
async def create_cloned_voice(
    voice_data: ClonedVoiceCreate,
    user: User = Depends(get_current_user),
):
    """Create a cloned voice from previously uploaded/downloaded audio.

    The audio file must already exist in the user's temp upload area
    (uploaded via /api/recording/upload or downloaded via /api/youtube/download).
    The transcript should come from /api/transcribe or manual entry.
    """
    # Find the audio file in temp area
    temp_dir = Path("/mnt/temp") / user.uid
    audio_files = list(temp_dir.glob("ref_*.wav")) + list(temp_dir.glob("recording_*.wav"))
    if not audio_files:
        raise HTTPException(
            status_code=400,
            detail="No audio file found. Upload audio or download from YouTube first.",
        )

    ref_audio = audio_files[-1]  # Use most recent

    voice = await create_voice(
        name=voice_data.name,
        user_id=user.uid,
        source_type=voice_data.source_type,
        language=voice_data.language,
        ref_audio_path=ref_audio,
        transcript=voice_data.transcript,
        model_id=voice_data.model_id,
        source_url=voice_data.source_url,
    )

    # Clean up temp files
    for f in audio_files:
        f.unlink(missing_ok=True)

    return voice


@router.delete("/cloned-voices/{voice_id}")
async def delete_cloned_voice(voice_id: str, user: User = Depends(get_current_user)):
    """Delete a cloned voice (creator only)."""
    if not await delete_voice(voice_id, user.uid):
        raise HTTPException(status_code=404, detail="Voice not found or not owned by you")
    return {"status": "deleted"}


@router.get("/cloned-voices/{voice_id}/sample")
async def get_voice_sample(voice_id: str, user: User = Depends(get_current_user)):
    """Generate a short TTS preview of a cloned voice."""
    voice = await get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    prompt_bytes = get_clone_prompt_bytes(voice_id)
    if not prompt_bytes:
        raise HTTPException(status_code=404, detail="Clone prompt not found")

    lang = iso_to_language(voice.language)
    wav_bytes = generate_sample(prompt_bytes, lang, voice.model_id)

    return Response(content=wav_bytes, media_type="audio/wav")


# ============================================================================
# YouTube Search & Download
# ============================================================================

@router.get("/youtube/search", response_model=list[YouTubeSearchResult])
async def youtube_search(q: str, user: User = Depends(get_current_user)):
    """Search YouTube for voice samples."""
    if not q or len(q) > 200:
        raise HTTPException(status_code=400, detail="Invalid search query")

    results = await search_youtube(f"{q} interview", max_results=5)
    return results


@router.post("/youtube/download")
async def youtube_download(
    req: YouTubeDownloadRequest,
    user: User = Depends(get_current_user),
):
    """Download and trim audio from a YouTube video."""
    temp_dir = Path("/mnt/temp") / user.uid
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_path = await download_audio(
        video_url=req.video_url,
        output_dir=temp_dir,
        max_duration_seconds=req.max_duration_seconds,
    )

    return {
        "status": "downloaded",
        "filename": audio_path.name,
        "duration_seconds": req.max_duration_seconds,
    }


# ============================================================================
# Recording Upload & Transcription
# ============================================================================

@router.post("/recording/upload")
async def upload_recording(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload an audio recording (from browser mic or file upload)."""
    temp_dir = Path("/mnt/temp") / user.uid
    temp_dir.mkdir(parents=True, exist_ok=True)

    import uuid as uuid_mod
    filename = f"recording_{uuid_mod.uuid4().hex}.wav"
    dest = temp_dir / filename

    with dest.open("wb") as f:
        content = await file.read()
        f.write(content)

    return {"status": "uploaded", "filename": filename}


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    language: str | None = None,
    user: User = Depends(get_current_user),
):
    """Transcribe the most recently uploaded/downloaded audio file."""
    temp_dir = Path("/mnt/temp") / user.uid
    audio_files = list(temp_dir.glob("ref_*.wav")) + list(temp_dir.glob("recording_*.wav"))
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio file found to transcribe")

    audio_path = audio_files[-1]
    result = transcribe(audio_path, language=language)

    return TranscriptionResult(**result)
```

**Step 2: Mount voice routes in main.py**

Modify `backend/main.py` — add the import and include the router alongside the existing one:

After `from api.routes import router` add:
```python
from api.voice_routes import router as voice_router
```

After `app.include_router(router)` add:
```python
app.include_router(voice_router)
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/api/voice_routes.py backend/main.py
git commit -m "$(cat <<'EOF'
feat: add API routes for voice cloning, YouTube search, transcription

Endpoints: cloned voice CRUD, YouTube search/download, recording upload,
Whisper transcription. All auth-protected via Authentik headers.
EOF
)"
```

---

### Task 7: Qwen3-TTS Conversion Pipeline

**Files:**
- Modify: `backend/services/converter.py`
- Modify: `backend/models/schemas.py`
- Modify: `backend/api/routes.py:133-158` (create_job)

**Step 1: Update Job schema to support cloned voices**

Modify `backend/models/schemas.py` — the `JobCreate` model needs to accept cloned voice IDs. The existing `voice` field already accepts any string, so no schema change is needed. But we should update the `Job` model to track voice type.

Add to `backend/models/schemas.py` after the `Job` class fields:
```python
    voice_type: str = "preset"  # "preset" (Kokoro) or "cloned" (Qwen3-TTS)
```

**Step 2: Add Qwen3-TTS pipeline to converter.py**

Modify `backend/services/converter.py`. Add imports at the top (after existing imports):

```python
from services.voice_cloner import is_cloned_voice, get_clone_prompt_bytes, get_voice
from services.qwen_tts import generate_chapter_audio, iso_to_language
```

Add a new method to `JobManager` (after `run_conversion`):

```python
    async def run_qwen_conversion(self, job_id: str):
        """Run ebook conversion using Qwen3-TTS with a cloned voice."""
        job = self.jobs.get(job_id)
        if not job:
            return

        try:
            self.update_job_status(job_id, JobStatus.PROCESSING)

            input_path = self.get_user_uploads_dir(job.user_id) / job.filename
            output_dir = self.get_user_outputs_dir(job.user_id) / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract chapters
            chapters: list[Chapter] = []
            if input_path.suffix.lower() == '.epub':
                chapters = extract_chapters(input_path)
                self.jobs[job_id].total_chapters = len(chapters)
                print(f"Extracted {len(chapters)} chapters for Qwen3-TTS conversion")

            if not chapters:
                self.update_job_status(job_id, JobStatus.FAILED, "No chapters found in EPUB")
                return

            # Load voice clone prompt
            clone_prompt = get_clone_prompt_bytes(job.voice)
            if not clone_prompt:
                self.update_job_status(job_id, JobStatus.FAILED, "Clone prompt not found for voice")
                return

            # Get voice metadata for language
            import asyncio
            voice_meta = await get_voice(job.voice)
            language = iso_to_language(voice_meta.language) if voice_meta else "English"
            model_id = voice_meta.model_id if voice_meta else None

            # Extract full text for each chapter
            from ebooklib import epub, ITEM_DOCUMENT
            from bs4 import BeautifulSoup

            book = epub.read_epub(str(input_path))
            chapter_texts = []
            for item in book.get_items():
                if item.get_type() != ITEM_DOCUMENT:
                    continue
                content = item.get_content()
                soup = BeautifulSoup(content, features='lxml')
                text_parts = []
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                    text = tag.get_text(strip=True)
                    if text:
                        text_parts.append(text)
                full_text = ' '.join(text_parts)
                filename = item.get_name() or ""
                from services.epub_parser import is_chapter as is_ch
                if is_ch(full_text, filename):
                    chapter_texts.append(full_text)

            # Generate audio per chapter
            book_name = input_path.stem
            wav_files = []

            for i, (chapter, text) in enumerate(zip(chapters, chapter_texts)):
                chapter_num = i + 1
                self.update_job_progress(
                    job_id,
                    (i / len(chapters)) * 90,
                    current_chapter=f"Generating chapter {chapter_num}/{len(chapters)}: {chapter.title}",
                )

                wav_path = output_dir / f"{book_name}_chapter_{chapter_num}.wav"

                try:
                    generate_chapter_audio(
                        chapter_text=text,
                        clone_prompt_bytes=clone_prompt,
                        output_path=wav_path,
                        language=language,
                        model_id=model_id,
                        speed=job.speed,
                    )
                    wav_files.append(wav_path)
                except torch.cuda.OutOfMemoryError:
                    self.update_job_status(
                        job_id, JobStatus.FAILED,
                        "GPU out of memory. Try the 0.6B model or wait for other jobs to finish.",
                    )
                    return
                except Exception as e:
                    print(f"Error generating chapter {chapter_num}: {e}")
                    self.update_job_status(job_id, JobStatus.FAILED, f"Chapter {chapter_num} failed: {e}")
                    return

            # Assemble M4B with FFmpeg
            self.update_job_progress(job_id, 90, current_chapter="Assembling audiobook...")

            m4b_path = output_dir / f"{book_name}.m4b"
            _assemble_m4b(wav_files, m4b_path)

            # Embed chapter metadata
            if chapters:
                try:
                    durations = get_chapter_audio_durations(output_dir)
                    if durations:
                        num = min(len(chapters), len(durations))
                        metadata = generate_ffmpeg_metadata(chapters[:num], durations[:num])
                        embed_chapters_in_m4b(m4b_path, metadata)

                        self.jobs[job_id].chapters = [
                            ChapterInfo(title=c.title, start_ms=c.start_ms, end_ms=c.end_ms)
                            for c in chapters[:num]
                        ]
                except Exception as e:
                    print(f"Failed to embed chapters: {e}")

            self.jobs[job_id].output_file = m4b_path.name
            self.update_job_status(job_id, JobStatus.COMPLETED)
            self.update_job_progress(job_id, 100.0)

        except Exception as e:
            print(f"Qwen3-TTS conversion error: {e}")
            self.update_job_status(job_id, JobStatus.FAILED, str(e))
```

Add a module-level helper function (outside the class, before the `job_manager` instance):

```python
def _assemble_m4b(wav_files: list[Path], output_path: Path):
    """Concatenate WAV files into an M4B using FFmpeg."""
    import tempfile

    # Create concat list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for wav in wav_files:
            f.write(f"file '{wav}'\n")
        concat_file = Path(f.name)

    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c:a', 'aac', '-b:a', '64k',
            '-movflags', '+faststart',
            str(output_path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        concat_file.unlink(missing_ok=True)
```

Also add `import torch` at the top of converter.py.

**Step 3: Update job creation route to route between pipelines**

Modify `backend/api/routes.py` — update the `create_job` endpoint (around line 133-158). Replace the voice validation and job start logic:

Replace the current `create_job` function with:

```python
@router.post("/jobs", response_model=Job)
async def create_job(job_create: JobCreate, user: User = Depends(get_current_user)):
    """Create a new conversion job."""
    from services.voice_cloner import is_cloned_voice

    # Verify file exists in user's uploads
    file_path = job_manager.get_user_uploads_dir(user.uid) / job_create.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Verify voice exists (preset or cloned)
    if is_cloned_voice(job_create.voice):
        from services.voice_cloner import get_voice as get_cloned
        voice = await get_cloned(job_create.voice)
        if not voice:
            raise HTTPException(status_code=404, detail="Cloned voice not found")
    else:
        voice = get_voice(job_create.voice)
        if not voice:
            raise HTTPException(status_code=404, detail="Voice not found")

    # Create job with user ownership
    job = job_manager.create_job(
        user_id=user.uid,
        filename=job_create.filename,
        voice=job_create.voice,
        speed=job_create.speed,
        use_gpu=job_create.use_gpu
    )

    # Route to correct pipeline
    if is_cloned_voice(job_create.voice):
        job.voice_type = "cloned"
        asyncio.create_task(job_manager.run_qwen_conversion(job.id))
    else:
        asyncio.create_task(job_manager.run_conversion(job.id))

    return job
```

**Step 4: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add backend/services/converter.py backend/models/schemas.py backend/api/routes.py
git commit -m "$(cat <<'EOF'
feat: add Qwen3-TTS conversion pipeline with dual routing

Converter routes between Kokoro CLI (preset voices) and Qwen3-TTS
Python pipeline (cloned voices) based on voice ID format.
Includes chapter-by-chapter generation, M4B assembly, and progress tracking.
EOF
)"
```

---

### Task 8: Frontend — VoicePicker Tabs

**Files:**
- Modify: `frontend/src/lib/VoicePicker.svelte`
- Modify: `frontend/src/App.svelte`

**Step 1: Add tabs to VoicePicker**

Replace `frontend/src/lib/VoicePicker.svelte` with the tabbed version. The existing preset voice list moves into a tab, and a new "Cloned Voices" tab is added alongside it.

The `selectedVoice` binding stays the same — it emits either a Kokoro voice ID (e.g., `af_sky`) or a cloned voice UUID.

Rewrite `VoicePicker.svelte`:

```svelte
<script>
  import ClonedVoicesList from './ClonedVoicesList.svelte';

  let { selectedVoice = $bindable(null) } = $props();

  let activeTab = $state('preset');
  let groupedVoices = $state({});
  let playingVoice = $state(null);
  let audioElement = $state(null);

  $effect(() => {
    fetchVoices();
  });

  async function fetchVoices() {
    try {
      const response = await fetch('/api/voices/grouped');
      if (response.ok) {
        groupedVoices = await response.json();
      }
    } catch (e) {
      console.error('Failed to fetch voices:', e);
    }
  }

  function selectVoice(voiceId) {
    selectedVoice = voiceId;
  }

  async function playVoiceSample(e, voiceId) {
    e.stopPropagation();

    if (playingVoice === voiceId) {
      if (audioElement) {
        audioElement.pause();
        audioElement = null;
      }
      playingVoice = null;
      return;
    }

    if (audioElement) {
      audioElement.pause();
    }

    playingVoice = voiceId;
    audioElement = new Audio(`/api/voices/${voiceId}/sample`);

    audioElement.onended = () => {
      playingVoice = null;
      audioElement = null;
    };

    audioElement.onerror = () => {
      playingVoice = null;
      audioElement = null;
    };

    try {
      await audioElement.play();
    } catch (err) {
      console.error('Failed to play sample:', err);
      playingVoice = null;
      audioElement = null;
    }
  }

  function getGenderIcon(gender) {
    return gender === 'F' ? '♀' : '♂';
  }

  function getLanguageFlag(language) {
    const flags = {
      'American English': '🇺🇸',
      'British English': '🇬🇧',
      'Japanese': '🇯🇵',
      'Mandarin Chinese': '🇨🇳',
      'Spanish': '🇪🇸',
      'French': '🇫🇷',
      'Hindi': '🇮🇳',
      'Italian': '🇮🇹',
      'Brazilian Portuguese': '🇧🇷'
    };
    return flags[language] || '🌐';
  }
</script>

<div class="voice-picker">
  <h3>Select Voice</h3>

  <div class="tabs">
    <button
      class="tab"
      class:active={activeTab === 'preset'}
      onclick={() => activeTab = 'preset'}
    >
      Preset Voices
    </button>
    <button
      class="tab"
      class:active={activeTab === 'cloned'}
      onclick={() => activeTab = 'cloned'}
    >
      Cloned Voices
    </button>
  </div>

  {#if activeTab === 'preset'}
    <div class="voice-groups">
      {#each Object.entries(groupedVoices) as [language, languageVoices]}
        <div class="voice-group">
          <div class="language-header">
            <span class="flag">{getLanguageFlag(language)}</span>
            <span class="language-name">{language}</span>
          </div>
          <div class="voices-list">
            {#each languageVoices as voice}
              <div
                class="voice-item"
                class:selected={selectedVoice === voice.id}
                onclick={() => selectVoice(voice.id)}
              >
                <button
                  class="play-btn"
                  class:playing={playingVoice === voice.id}
                  onclick={(e) => playVoiceSample(e, voice.id)}
                  title="Play sample"
                >
                  {playingVoice === voice.id ? '⏹' : '▶'}
                </button>
                <span class="voice-name">{voice.name}</span>
                <span class="voice-id">{voice.id}</span>
                <span class="voice-gender" class:female={voice.gender === 'F'}>
                  {getGenderIcon(voice.gender)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <ClonedVoicesList bind:selectedVoice />
  {/if}
</div>

<style>
  .voice-picker {
    max-height: 400px;
    overflow-y: auto;
  }

  h3 {
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: #333;
    position: sticky;
    top: 0;
    background: white;
    padding: 0.5rem 0;
    z-index: 1;
  }

  .tabs {
    display: flex;
    gap: 0;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e0e0e0;
    position: sticky;
    top: 2.5rem;
    background: white;
    z-index: 1;
  }

  .tab {
    padding: 0.5rem 1rem;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    color: #666;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.2s, border-color 0.2s;
  }

  .tab:hover {
    color: #333;
  }

  .tab.active {
    color: #4a90d9;
    border-bottom-color: #4a90d9;
  }

  .voice-groups {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .voice-group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
  }

  .language-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: #f5f5f5;
    font-weight: 500;
  }

  .flag {
    font-size: 1.25rem;
  }

  .voices-list {
    display: flex;
    flex-direction: column;
  }

  .voice-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: background 0.2s;
  }

  .voice-item:hover {
    background: #f0f7ff;
  }

  .voice-item.selected {
    background: #e3f2fd;
    border-left: 3px solid #4a90d9;
  }

  .voice-name {
    flex: 1;
    font-weight: 500;
  }

  .voice-id {
    font-size: 0.75rem;
    color: #888;
    font-family: monospace;
  }

  .voice-gender {
    font-size: 1rem;
    color: #2196f3;
  }

  .voice-gender.female {
    color: #e91e63;
  }

  .play-btn {
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 50%;
    background: #4a90d9;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    transition: background 0.2s, transform 0.1s;
    flex-shrink: 0;
  }

  .play-btn:hover {
    background: #3a7bc8;
    transform: scale(1.1);
  }

  .play-btn.playing {
    background: #e91e63;
  }

  .play-btn.playing:hover {
    background: #c2185b;
  }
</style>
```

**Step 2: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add frontend/src/lib/VoicePicker.svelte
git commit -m "$(cat <<'EOF'
feat: add Preset/Cloned tabs to VoicePicker

Tabbed UI separating Kokoro preset voices from cloned voices.
Preset tab unchanged, cloned tab delegates to ClonedVoicesList.
EOF
)"
```

---

### Task 9: Frontend — ClonedVoicesList Component

**Files:**
- Create: `frontend/src/lib/ClonedVoicesList.svelte`

**Step 1: Create the component**

Create `frontend/src/lib/ClonedVoicesList.svelte`:

```svelte
<script>
  import CreateVoiceModal from './CreateVoiceModal.svelte';

  let { selectedVoice = $bindable(null) } = $props();

  let voices = $state([]);
  let loading = $state(true);
  let showCreateModal = $state(false);
  let searchQuery = $state('');
  let playingVoice = $state(null);
  let audioElement = $state(null);

  $effect(() => {
    fetchVoices();
  });

  async function fetchVoices() {
    loading = true;
    try {
      const response = await fetch('/api/cloned-voices');
      if (response.ok) {
        voices = await response.json();
      }
    } catch (e) {
      console.error('Failed to fetch cloned voices:', e);
    } finally {
      loading = false;
    }
  }

  let filteredVoices = $derived(
    voices.filter(v =>
      v.name.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  function selectVoice(voiceId) {
    selectedVoice = voiceId;
  }

  async function playVoiceSample(e, voiceId) {
    e.stopPropagation();

    if (playingVoice === voiceId) {
      if (audioElement) {
        audioElement.pause();
        audioElement = null;
      }
      playingVoice = null;
      return;
    }

    if (audioElement) {
      audioElement.pause();
    }

    playingVoice = voiceId;
    audioElement = new Audio(`/api/cloned-voices/${voiceId}/sample`);

    audioElement.onended = () => {
      playingVoice = null;
      audioElement = null;
    };

    audioElement.onerror = () => {
      playingVoice = null;
      audioElement = null;
    };

    try {
      await audioElement.play();
    } catch (err) {
      console.error('Failed to play sample:', err);
      playingVoice = null;
      audioElement = null;
    }
  }

  async function deleteVoice(e, voiceId) {
    e.stopPropagation();
    if (!confirm('Delete this cloned voice? This cannot be undone.')) return;

    try {
      const res = await fetch(`/api/cloned-voices/${voiceId}`, { method: 'DELETE' });
      if (res.ok) {
        await fetchVoices();
        if (selectedVoice === voiceId) selectedVoice = null;
      }
    } catch (e) {
      console.error('Failed to delete voice:', e);
    }
  }

  function handleVoiceCreated() {
    showCreateModal = false;
    fetchVoices();
  }

  function getSourceBadge(sourceType) {
    const badges = {
      youtube: 'YT',
      upload: 'UP',
      recording: 'MIC',
      curated: '★',
    };
    return badges[sourceType] || sourceType;
  }

  function getLanguageFlag(lang) {
    const flags = {
      en: '🇺🇸', zh: '🇨🇳', ja: '🇯🇵', ko: '🇰🇷',
      de: '🇩🇪', fr: '🇫🇷', ru: '🇷🇺', pt: '🇧🇷',
      es: '🇪🇸', it: '🇮🇹',
    };
    return flags[lang] || '🌐';
  }
</script>

<div class="cloned-voices">
  <div class="search-bar">
    <input
      type="text"
      placeholder="Search cloned voices..."
      bind:value={searchQuery}
    />
  </div>

  {#if loading}
    <p class="empty">Loading...</p>
  {:else if filteredVoices.length === 0}
    <p class="empty">
      {searchQuery ? 'No matching voices' : 'No cloned voices yet'}
    </p>
  {:else}
    <div class="voices-list">
      {#each filteredVoices as voice}
        <div
          class="voice-item"
          class:selected={selectedVoice === voice.id}
          onclick={() => selectVoice(voice.id)}
        >
          <button
            class="play-btn"
            class:playing={playingVoice === voice.id}
            onclick={(e) => playVoiceSample(e, voice.id)}
            title="Play sample"
          >
            {playingVoice === voice.id ? '⏹' : '▶'}
          </button>
          <span class="voice-name">{voice.name}</span>
          <span class="voice-lang">{getLanguageFlag(voice.language)}</span>
          <span class="voice-source">{getSourceBadge(voice.source_type)}</span>
          <button
            class="delete-btn"
            onclick={(e) => deleteVoice(e, voice.id)}
            title="Delete voice"
          >
            ×
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <button class="create-btn" onclick={() => showCreateModal = true}>
    + Create New Voice
  </button>

  {#if showCreateModal}
    <CreateVoiceModal
      onClose={() => showCreateModal = false}
      onCreated={handleVoiceCreated}
    />
  {/if}
</div>

<style>
  .cloned-voices {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .search-bar input {
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 0.875rem;
    box-sizing: border-box;
  }

  .search-bar input:focus {
    outline: none;
    border-color: #4a90d9;
  }

  .empty {
    color: #666;
    text-align: center;
    padding: 1.5rem;
    font-size: 0.875rem;
  }

  .voices-list {
    display: flex;
    flex-direction: column;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
  }

  .voice-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: background 0.2s;
    border-bottom: 1px solid #f0f0f0;
  }

  .voice-item:last-child {
    border-bottom: none;
  }

  .voice-item:hover {
    background: #f0f7ff;
  }

  .voice-item.selected {
    background: #e3f2fd;
    border-left: 3px solid #4a90d9;
  }

  .voice-name {
    flex: 1;
    font-weight: 500;
    font-size: 0.875rem;
  }

  .voice-lang {
    font-size: 1rem;
  }

  .voice-source {
    font-size: 0.625rem;
    color: #666;
    background: #f0f0f0;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .play-btn {
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 50%;
    background: #4a90d9;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
  }

  .play-btn:hover { background: #3a7bc8; }
  .play-btn.playing { background: #e91e63; }

  .delete-btn {
    width: 24px;
    height: 24px;
    border: none;
    background: none;
    color: #999;
    cursor: pointer;
    font-size: 1rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .delete-btn:hover {
    background: #ffebee;
    color: #f44336;
  }

  .create-btn {
    padding: 0.625rem 1rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .create-btn:hover {
    background: #3a7fc9;
  }
</style>
```

**Step 2: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add frontend/src/lib/ClonedVoicesList.svelte
git commit -m "$(cat <<'EOF'
feat: add ClonedVoicesList component

Searchable list of cloned voices with play preview, delete, and
create new voice button. Integrates with VoicePicker tabs.
EOF
)"
```

---

### Task 10: Frontend — CreateVoiceModal (YouTube, Upload, Record)

**Files:**
- Create: `frontend/src/lib/CreateVoiceModal.svelte`
- Create: `frontend/src/lib/AudioRecorder.svelte`

**Step 1: Create AudioRecorder component**

Create `frontend/src/lib/AudioRecorder.svelte`:

```svelte
<script>
  let { onRecorded = () => {} } = $props();

  let recording = $state(false);
  let mediaRecorder = $state(null);
  let audioChunks = $state([]);
  let recordedUrl = $state(null);
  let error = $state(null);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' });
        recordedUrl = URL.createObjectURL(blob);

        // Upload to server
        const formData = new FormData();
        formData.append('file', blob, 'recording.wav');

        try {
          const res = await fetch('/api/recording/upload', {
            method: 'POST',
            body: formData,
          });
          if (res.ok) {
            onRecorded();
          } else {
            error = 'Upload failed';
          }
        } catch (e) {
          error = 'Upload failed: ' + e.message;
        }

        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      recording = true;
      error = null;
    } catch (e) {
      error = 'Microphone access denied';
    }
  }

  function stopRecording() {
    if (mediaRecorder && recording) {
      mediaRecorder.stop();
      recording = false;
    }
  }
</script>

<div class="recorder">
  {#if !recordedUrl}
    <button
      class="record-btn"
      class:recording
      onclick={recording ? stopRecording : startRecording}
    >
      {recording ? '⏹ Stop Recording' : '🎤 Start Recording'}
    </button>
    {#if recording}
      <span class="recording-indicator">Recording...</span>
    {/if}
  {:else}
    <div class="recorded">
      <audio controls src={recordedUrl}></audio>
      <button class="redo-btn" onclick={() => { recordedUrl = null; }}>
        Re-record
      </button>
    </div>
  {/if}

  {#if error}
    <p class="error">{error}</p>
  {/if}
</div>

<style>
  .recorder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
  }

  .record-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    background: #4a90d9;
    color: white;
    transition: background 0.2s;
  }

  .record-btn.recording {
    background: #f44336;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .recording-indicator {
    color: #f44336;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .recorded {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .recorded audio {
    max-width: 100%;
  }

  .redo-btn {
    padding: 0.375rem 0.75rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .error {
    color: #f44336;
    font-size: 0.875rem;
  }
</style>
```

**Step 2: Create CreateVoiceModal component**

Create `frontend/src/lib/CreateVoiceModal.svelte`:

```svelte
<script>
  import AudioRecorder from './AudioRecorder.svelte';

  let { onClose = () => {}, onCreated = () => {} } = $props();

  let mode = $state('search'); // 'search' | 'upload' | 'record'
  let step = $state('source'); // 'source' | 'transcript' | 'create'

  // Voice metadata
  let voiceName = $state('');
  let language = $state('en');
  let modelId = $state('Qwen/Qwen3-TTS-12Hz-1.7B-Base');
  let transcript = $state('');
  let sourceUrl = $state('');
  let models = $state([]);

  // YouTube search
  let searchQuery = $state('');
  let searchResults = $state([]);
  let searching = $state(false);
  let downloading = $state(false);

  // Upload
  let uploading = $state(false);

  // Transcription
  let transcribing = $state(false);

  // Creation
  let creating = $state(false);
  let error = $state(null);

  let hasAudio = $state(false);

  $effect(() => {
    fetchModels();
  });

  async function fetchModels() {
    try {
      const res = await fetch('/api/cloned-voices/models');
      if (res.ok) models = await res.json();
    } catch (e) {}
  }

  async function searchYouTube() {
    if (!searchQuery.trim()) return;
    searching = true;
    error = null;
    try {
      const res = await fetch(`/api/youtube/search?q=${encodeURIComponent(searchQuery)}`);
      if (res.ok) {
        searchResults = await res.json();
        if (searchResults.length === 0) error = 'No results found';
      } else {
        error = 'Search failed';
      }
    } catch (e) {
      error = 'Search failed: ' + e.message;
    } finally {
      searching = false;
    }
  }

  async function downloadVideo(result) {
    downloading = true;
    error = null;
    sourceUrl = result.url;
    if (!voiceName) voiceName = searchQuery;
    try {
      const res = await fetch('/api/youtube/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: result.url }),
      });
      if (res.ok) {
        hasAudio = true;
        await autoTranscribe();
      } else {
        const data = await res.json();
        error = data.detail || 'Download failed';
      }
    } catch (e) {
      error = 'Download failed: ' + e.message;
    } finally {
      downloading = false;
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    uploading = true;
    error = null;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/recording/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        hasAudio = true;
        await autoTranscribe();
      } else {
        error = 'Upload failed';
      }
    } catch (e) {
      error = 'Upload failed: ' + e.message;
    } finally {
      uploading = false;
    }
  }

  function handleRecorded() {
    hasAudio = true;
    autoTranscribe();
  }

  async function autoTranscribe() {
    transcribing = true;
    step = 'transcript';
    try {
      const res = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (res.ok) {
        const data = await res.json();
        transcript = data.text;
        if (data.language) language = data.language;
      }
    } catch (e) {
      console.error('Transcription error:', e);
    } finally {
      transcribing = false;
    }
  }

  async function createVoice() {
    if (!voiceName.trim() || !transcript.trim()) {
      error = 'Voice name and transcript are required';
      return;
    }

    creating = true;
    error = null;
    try {
      const res = await fetch('/api/cloned-voices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: voiceName,
          language,
          model_id: modelId,
          transcript,
          source_type: mode === 'search' ? 'youtube' : mode === 'upload' ? 'upload' : 'recording',
          source_url: sourceUrl || null,
        }),
      });

      if (res.ok) {
        onCreated();
      } else {
        const data = await res.json();
        error = data.detail || 'Failed to create voice';
      }
    } catch (e) {
      error = 'Failed to create voice: ' + e.message;
    } finally {
      creating = false;
    }
  }

  function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'de', name: 'German' },
    { code: 'fr', name: 'French' },
    { code: 'ru', name: 'Russian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'es', name: 'Spanish' },
    { code: 'it', name: 'Italian' },
  ];
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={onClose}>
  <div class="modal" onclick|stopPropagation>
    <div class="modal-header">
      <h2>Create Voice Clone</h2>
      <button class="close-btn" onclick={onClose}>×</button>
    </div>

    {#if step === 'source'}
      <div class="mode-tabs">
        <button class:active={mode === 'search'} onclick={() => mode = 'search'}>
          🔍 Search by Name
        </button>
        <button class:active={mode === 'upload'} onclick={() => mode = 'upload'}>
          📁 Upload File
        </button>
        <button class:active={mode === 'record'} onclick={() => mode = 'record'}>
          🎤 Record Voice
        </button>
      </div>

      <div class="mode-content">
        {#if mode === 'search'}
          <div class="search-section">
            <div class="search-input">
              <input
                type="text"
                placeholder="Person's name (e.g., Morgan Freeman)"
                bind:value={searchQuery}
                onkeydown={(e) => e.key === 'Enter' && searchYouTube()}
              />
              <button onclick={searchYouTube} disabled={searching || !searchQuery.trim()}>
                {searching ? 'Searching...' : 'Search'}
              </button>
            </div>

            {#if searchResults.length > 0}
              <div class="results">
                {#each searchResults as result}
                  <div class="result-item">
                    <div class="result-info">
                      <span class="result-title">{result.title}</span>
                      <span class="result-duration">{formatDuration(result.duration_seconds)}</span>
                    </div>
                    <button
                      class="select-btn"
                      onclick={() => downloadVideo(result)}
                      disabled={downloading}
                    >
                      {downloading ? 'Downloading...' : 'Select'}
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

        {:else if mode === 'upload'}
          <div class="upload-section">
            <label class="file-upload">
              {uploading ? 'Uploading...' : 'Choose audio file (WAV, MP3)'}
              <input
                type="file"
                accept="audio/*"
                onchange={handleFileUpload}
                disabled={uploading}
              />
            </label>
          </div>

        {:else if mode === 'record'}
          <AudioRecorder onRecorded={handleRecorded} />
        {/if}
      </div>

    {:else if step === 'transcript'}
      <div class="transcript-section">
        <label>
          Transcript
          {#if transcribing}
            <span class="transcribing">(auto-transcribing...)</span>
          {/if}
        </label>
        <textarea
          bind:value={transcript}
          rows="4"
          placeholder="Edit the transcript if needed..."
        ></textarea>

        <div class="form-row">
          <label>
            Voice Name
            <input type="text" bind:value={voiceName} placeholder="e.g., Morgan Freeman" />
          </label>
        </div>

        <div class="form-row">
          <label>
            Language
            <select bind:value={language}>
              {#each languages as lang}
                <option value={lang.code}>{lang.name}</option>
              {/each}
            </select>
          </label>
        </div>

        <div class="form-row">
          <label>
            Model
            <select bind:value={modelId}>
              {#each models as model}
                <option value={model.id}>
                  {model.name} (~{model.vram_gb} GB VRAM)
                </option>
              {/each}
            </select>
          </label>
        </div>

        <div class="actions">
          <button class="back-btn" onclick={() => { step = 'source'; hasAudio = false; }}>
            Back
          </button>
          <button
            class="create-btn"
            onclick={createVoice}
            disabled={creating || !voiceName.trim() || !transcript.trim()}
          >
            {creating ? 'Creating...' : 'Create Voice'}
          </button>
        </div>
      </div>
    {/if}

    {#if error}
      <p class="error">{error}</p>
    {/if}
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: white;
    border-radius: 12px;
    width: 90%;
    max-width: 550px;
    max-height: 80vh;
    overflow-y: auto;
    padding: 1.5rem;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .modal-header h2 {
    margin: 0;
    font-size: 1.25rem;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: #666;
  }

  .mode-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .mode-tabs button {
    flex: 1;
    padding: 0.5rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 0.8rem;
    transition: all 0.2s;
  }

  .mode-tabs button.active {
    background: #e3f2fd;
    border-color: #4a90d9;
    color: #4a90d9;
  }

  .search-input {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .search-input input {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 0.875rem;
  }

  .search-input button {
    padding: 0.5rem 1rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
  }

  .search-input button:disabled {
    background: #ccc;
  }

  .results {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .result-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.625rem 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
  }

  .result-info {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
  }

  .result-title {
    font-size: 0.875rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .result-duration {
    font-size: 0.75rem;
    color: #666;
  }

  .select-btn {
    padding: 0.375rem 0.75rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-left: 0.5rem;
  }

  .select-btn:disabled {
    background: #ccc;
  }

  .upload-section {
    text-align: center;
    padding: 2rem 1rem;
  }

  .file-upload {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    background: #4a90d9;
    color: white;
    border-radius: 8px;
    cursor: pointer;
  }

  .file-upload input {
    display: none;
  }

  .transcript-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .transcript-section label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #333;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .transcribing {
    font-weight: normal;
    color: #4a90d9;
    font-size: 0.8rem;
  }

  .transcript-section textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 0.875rem;
    font-family: inherit;
    resize: vertical;
    box-sizing: border-box;
  }

  .transcript-section input,
  .transcript-section select {
    padding: 0.5rem 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 0.875rem;
  }

  .form-row {
    display: flex;
    flex-direction: column;
  }

  .actions {
    display: flex;
    gap: 0.75rem;
    justify-content: flex-end;
    margin-top: 0.5rem;
  }

  .back-btn {
    padding: 0.5rem 1rem;
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    cursor: pointer;
  }

  .create-btn {
    padding: 0.5rem 1.5rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
  }

  .create-btn:disabled {
    background: #ccc;
  }

  .error {
    color: #f44336;
    font-size: 0.875rem;
    margin-top: 0.5rem;
    text-align: center;
  }
</style>
```

**Step 3: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add frontend/src/lib/CreateVoiceModal.svelte frontend/src/lib/AudioRecorder.svelte
git commit -m "$(cat <<'EOF'
feat: add CreateVoiceModal and AudioRecorder components

Modal with three input modes: YouTube search, file upload, mic recording.
Auto-transcribes with Whisper, lets user edit transcript and select
model/language before creating the voice clone.
EOF
)"
```

---

### Task 11: Dockerfile Updates

**Files:**
- Modify: `Dockerfile`

**Step 1: Add new pip dependencies to Dockerfile**

After the existing `pip install` line in the Dockerfile, add the new dependencies and environment variables.

Add after the `RUN pip install --no-cache-dir --break-system-packages -r requirements.txt` line:
```dockerfile
RUN pip install --no-cache-dir --break-system-packages qwen-tts openai-whisper yt-dlp
```

Add before the `EXPOSE 8000` line:
```dockerfile
ENV QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV WHISPER_MODEL=small
ENV VOICES_DB=/mnt/voices/voices.db
```

**Step 2: Commit**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add Dockerfile
git commit -m "$(cat <<'EOF'
feat: add qwen-tts, whisper, and yt-dlp to Docker image

Adds voice cloning dependencies and environment variable defaults
for model selection and database path.
EOF
)"
```

---

### Task 12: Integration Testing & Smoke Test

**Files:**
- All files modified in tasks 1-11

**Step 1: Verify backend imports**

Run from `backend/` directory to check all imports resolve:

```bash
cd /Users/viktorbarzin/code/audiblez-web/backend
python -c "
from services.database import init_db, execute_read, execute_write
from services.youtube_search import search_youtube, download_audio
from services.transcriber import transcribe
from services.voice_cloner import list_voices, create_voice, delete_voice, is_cloned_voice
from models.voice_schemas import ClonedVoice, ClonedVoiceCreate, YouTubeSearchResult
from api.voice_routes import router
print('All imports OK')
"
```

Expected: `All imports OK` (or import errors for qwen-tts/whisper since those aren't installed locally — that's fine, they'll work in Docker).

**Step 2: Verify frontend builds**

```bash
cd /Users/viktorbarzin/code/audiblez-web/frontend
npm install && npm run build
```

Expected: Build succeeds with no errors. Warnings are acceptable.

**Step 3: Verify database init works locally**

```bash
cd /Users/viktorbarzin/code/audiblez-web/backend
VOICES_DB=/tmp/test_voices.db python -c "
from services.database import init_db, execute_read
init_db()
import asyncio
rows = asyncio.run(execute_read('SELECT name FROM sqlite_master WHERE type=\"table\"'))
print('Tables:', [r['name'] for r in rows])
assert any(r['name'] == 'voices' for r in rows), 'voices table not found'
print('Database init OK')
"
rm /tmp/test_voices.db
```

Expected: `Tables: ['voices']` and `Database init OK`

**Step 4: Commit any fixes**

If steps 1-3 revealed any issues, fix them and commit:

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add -A
git commit -m "fix: resolve integration issues from smoke testing"
```

---

### Task 13: Docker Build Test

**Step 1: Build the Docker image**

```bash
cd /Users/viktorbarzin/code/audiblez-web
docker build -t audiblez-web:voice-cloning .
```

Expected: Build completes successfully. This will verify all pip dependencies install correctly in the container environment.

**Step 2: Quick container smoke test**

```bash
docker run --rm -it -p 8000:8000 \
  -e VOICES_DB=/tmp/voices.db \
  -e QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  audiblez-web:voice-cloning \
  python -c "from services.database import init_db; init_db(); print('Container OK')"
```

**Step 3: Commit if any Dockerfile fixes needed**

```bash
cd /Users/viktorbarzin/code/audiblez-web
git add Dockerfile
git commit -m "fix: docker build fixes for voice cloning dependencies"
```
