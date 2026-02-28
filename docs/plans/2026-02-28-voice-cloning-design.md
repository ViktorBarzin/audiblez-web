# Voice Cloning Design — Qwen3-TTS Integration

**Date:** 2026-02-28
**Status:** Approved

## Overview

Add voice cloning capability to audiblez-web using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Users can clone any voice from a 3-second audio sample and use it for full ebook-to-audiobook conversion. Three ways to provide reference audio: search by name (YouTube), upload a file, or record via browser microphone.

## Architecture

```
+-------------------------------------------------------------+
|                   audiblez-web container                      |
|                                                              |
|  +---------------+    +----------------------------------+   |
|  |  Svelte 5 UI  |--->|  FastAPI Backend (port 8000)     |   |
|  |               |    |                                  |   |
|  | - Upload EPUB |    |  /api/cloned-voices  (new)       |   |
|  | - Pick voice  |    |  /api/youtube/search (new)       |   |
|  | - Clone mgmt  |    |  /api/transcribe     (new)       |   |
|  | - Record mic  |    |  /api/jobs           (existing)  |   |
|  +---------------+    |                                  |   |
|                       |  +----------------------------+  |   |
|                       |  |   Conversion Router        |  |   |
|                       |  |                            |  |   |
|                       |  |  Kokoro voice?             |  |   |
|                       |  |   -> audiblez CLI          |  |   |
|                       |  |                            |  |   |
|                       |  |  Cloned voice?             |  |   |
|                       |  |   -> Qwen3-TTS pipeline    |  |   |
|                       |  +----------------------------+  |   |
|                       |                                  |   |
|                       |  +--------+ +--------+ +------+  |   |
|                       |  |Qwen3   | |Whisper | |yt-   |  |   |
|                       |  |TTS     | |(STT)   | |dlp   |  |   |
|                       |  +--------+ +--------+ +------+  |   |
|                       +----------------------------------+   |
|                                                              |
|  +----------+  +-----------------------------------------+   |
|  | SQLite   |  | NFS Storage                             |   |
|  | voices.db|  | /mnt/voices/{id}/ref.wav                |   |
|  |          |  | /mnt/voices/{id}/clone_prompt.bin        |   |
|  |          |  | /mnt/users/{uid}/outputs/...             |   |
|  +----------+  +-----------------------------------------+   |
+-------------------------------------------------------------+
```

### Key Decisions

- **Dual pipeline**: Kokoro preset voices use the existing audiblez CLI (unchanged). Cloned voices use a new Python-native pipeline calling Qwen3-TTS directly.
- **Embedded**: Qwen3-TTS, Whisper, and yt-dlp are pip-installed into the existing container rather than deployed as separate services.
- **SQLite**: Voice profiles stored in SQLite (WAL mode) on NFS. No new infrastructure.
- **Shared voices**: All cloned voices are public by default — any authenticated user can use any voice.

## Voice Cloning Workflows

### Path 1: Search by Name (YouTube)

1. User types a person's name (e.g., "Morgan Freeman")
2. Backend searches YouTube via yt-dlp: `"{name} interview"`
3. Returns top 5 results with titles and durations
4. User selects a video
5. yt-dlp downloads audio, trimmed to first 30 seconds
6. Whisper transcribes the audio
7. User reviews and optionally edits the transcript
8. Qwen3-TTS creates a clone prompt from audio + transcript
9. Voice profile saved to SQLite + reference audio/clone prompt to NFS

### Path 2: Pre-curated Library

1. User browses curated voices (shipped with the container)
2. Pre-built clone prompts already stored
3. User selects a voice — immediately available for conversion

### Path 3: User-provided Audio

1. User uploads a WAV/MP3 file OR records via browser microphone
2. Whisper auto-transcribes (user can edit)
3. Qwen3-TTS creates clone prompt
4. User names the voice and selects language
5. Voice profile saved

## Data Model

### SQLite Schema

```sql
CREATE TABLE voices (
    id          TEXT PRIMARY KEY,    -- UUID
    name        TEXT NOT NULL,       -- Display name, e.g., "Morgan Freeman"
    created_by  TEXT NOT NULL,       -- Authentik user_id
    source_type TEXT NOT NULL,       -- 'youtube' | 'upload' | 'recording' | 'curated'
    source_url  TEXT,                -- YouTube URL if applicable
    language    TEXT NOT NULL,       -- ISO 639-1 code: 'en', 'zh', etc.
    ref_audio   TEXT NOT NULL,       -- Path to reference WAV on NFS
    transcript  TEXT NOT NULL,       -- Transcript of reference audio
    clone_prompt BLOB,              -- Serialized Qwen3-TTS clone prompt
    model_id    TEXT NOT NULL,       -- e.g., 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
    is_public   BOOLEAN DEFAULT 1,  -- Visible to all users
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### NFS Storage Layout

```
/mnt/voices/
  {voice_id}/
    ref.wav              -- Reference audio (3-30 seconds)
    clone_prompt.bin     -- Serialized Qwen3-TTS prompt (reusable)
```

## Qwen3-TTS Conversion Pipeline

When a cloned voice is selected for ebook conversion:

1. **EPUB parsing** — existing `epub_parser.py` extracts chapters
2. **Load clone prompt** — from NFS (created during voice setup)
3. **Per-chapter generation** — sequential, one chapter at a time:
   - Long chapters split into ~500-word segments at sentence boundaries
   - `model.generate_voice_clone(text=segment, voice_clone_prompt=prompt)`
   - Segments concatenated with pydub
   - Progress reported via WebSocket (same interface as Kokoro jobs)
4. **Assembly** — existing `chapter_embedder.py` creates M4B with chapter metadata via FFmpeg
5. **Speed adjustment** — applied via pydub resampling (Qwen3-TTS has no native speed control)

### Model Loading

- Qwen3-TTS loaded lazily on first cloned-voice operation
- Stays in GPU memory between jobs (avoid ~30s reload)
- Model configurable via `QWEN_TTS_MODEL` environment variable

### VRAM Budget (Tesla T4, 16 GB)

| Component | VRAM |
|-----------|------|
| Qwen3-TTS 0.6B-Base (bfloat16) | ~2 GB |
| Qwen3-TTS 1.7B-Base (bfloat16) | ~5 GB |
| Whisper small | ~1 GB |
| Kokoro-82M (via audiblez) | ~1 GB |
| CUDA overhead | ~1 GB |
| **Total (0.6B config)** | **~5 GB** |
| **Total (1.7B config)** | **~8 GB** |

## API Endpoints

### Voice Management (auth required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cloned-voices` | List all public + user's cloned voices |
| POST | `/api/cloned-voices` | Create voice clone |
| GET | `/api/cloned-voices/{id}` | Get voice details |
| DELETE | `/api/cloned-voices/{id}` | Delete voice (creator only) |
| GET | `/api/cloned-voices/{id}/sample` | Generate short TTS preview |

### YouTube Search (auth required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/youtube/search?q=...` | Search YouTube, return top 5 |
| POST | `/api/youtube/download` | Download + trim audio from video |

### Recording & Transcription (auth required)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/recording/upload` | Upload recorded audio blob |
| POST | `/api/transcribe` | Whisper transcribe audio file |

## UI Changes

### Modified: VoicePicker.svelte

Add tabs: `[Preset Voices]` and `[Cloned Voices]`. Preset tab is unchanged. Cloned tab shows searchable list of cloned voices with play button, creator badge, and a `[+ Create New Voice]` button.

### New: CreateVoiceModal.svelte

Modal with three input modes (Search by Name / Upload File / Record Voice). Contains:
- YouTube search results with audio preview
- File upload dropzone
- Browser microphone recorder (MediaRecorder API)
- Auto-generated transcript with edit capability
- Voice name input
- Language selector
- Model selector showing VRAM usage per option

### New Components

- `ClonedVoicesList.svelte` — cloned voices tab content
- `YouTubeSearch.svelte` — search + select video
- `AudioRecorder.svelte` — browser microphone recording
- `TranscriptEditor.svelte` — editable transcript
- `ModelSelector.svelte` — model picker with VRAM info

## Backend File Structure

### New Files

```
backend/services/
  qwen_tts.py          -- Model loading, generation, clone prompt creation
  voice_cloner.py      -- Voice CRUD, clone workflow orchestration
  youtube_search.py    -- yt-dlp search + audio download
  transcriber.py       -- Whisper transcription
  database.py          -- SQLite setup, connection, migrations

backend/api/
  voice_routes.py      -- Cloned voice API endpoints
```

### Modified Files

```
backend/services/converter.py  -- Add routing: Kokoro vs Qwen3-TTS
backend/api/routes.py          -- Mount voice_routes
backend/main.py                -- Initialize DB on startup
frontend/src/lib/VoicePicker.svelte -- Add tabs
```

## Dockerfile Changes

```dockerfile
# Added to existing Dockerfile
RUN pip install qwen-tts openai-whisper yt-dlp pydub

ENV QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV WHISPER_MODEL=small
ENV VOICES_DB=/mnt/voices/voices.db
```

Models auto-download on first use, cached on NFS.

## Error Handling

| Scenario | Response |
|----------|----------|
| YouTube search returns no results | Empty list, UI shows "No results" |
| yt-dlp download fails | 502, suggest upload instead |
| Whisper transcription fails | Return partial transcript, allow manual edit |
| Clone prompt creation fails | 500, suggest different audio sample |
| CUDA OOM during conversion | 503 "GPU busy", suggest 0.6B model |
| Long chapter exceeds context | Split at sentence boundaries (~500 words) |
| SQLite concurrent writes | WAL mode + asyncio lock for serialization |

## Kubernetes Changes

None. Same container, same port, same NFS mounts. The only implicit requirement is internet access for yt-dlp (already available for model downloads).
