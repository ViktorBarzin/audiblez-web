"""Voice cloner service with CRUD operations.

Manages cloned voices on NFS storage and in the SQLite database.
Each cloned voice gets a directory under ``/mnt/voices/{voice_id}/``
containing the reference audio (``ref.wav``) and the serialised clone
prompt (``clone_prompt.bin``).
"""

import logging
import shutil
import uuid
from pathlib import Path

from models.voice_schemas import ClonedVoice
from services.database import execute_read, execute_write
from services.qwen_tts import create_clone_prompt, iso_to_language

logger = logging.getLogger(__name__)

VOICES_DIR = Path("/mnt/voices")


def _voice_dir(voice_id: str) -> Path:
    """Return (and ensure existence of) the NFS directory for a voice."""
    d = VOICES_DIR / voice_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def list_voices(user_id: str) -> list[ClonedVoice]:
    """Return all public voices and the caller's private voices."""
    rows = await execute_read(
        "SELECT * FROM voices WHERE is_public = 1 OR created_by = ? "
        "ORDER BY created_at DESC",
        (user_id,),
    )
    return [ClonedVoice(**row) for row in rows]


async def get_voice(voice_id: str) -> ClonedVoice | None:
    """Look up a single voice by its ID."""
    rows = await execute_read(
        "SELECT * FROM voices WHERE id = ?", (voice_id,)
    )
    if rows:
        return ClonedVoice(**rows[0])
    return None


async def create_voice(
    name: str,
    user_id: str,
    source_type: str,
    language: str,
    ref_audio_path: str | Path,
    transcript: str,
    model_id: str,
    source_url: str | None = None,
) -> ClonedVoice:
    """Create a new cloned voice.

    1. Copy the reference audio into the NFS voice directory.
    2. Generate the clone prompt via Qwen3-TTS.
    3. Persist the clone prompt binary to NFS.
    4. Insert a row into the ``voices`` table.
    5. Return the newly created :class:`ClonedVoice`.
    """
    voice_id = str(uuid.uuid4())
    vdir = _voice_dir(voice_id)

    # 1. Copy reference audio to NFS
    ref_dest = vdir / "ref.wav"
    shutil.copy2(str(ref_audio_path), str(ref_dest))
    logger.info("Copied ref audio to %s", ref_dest)

    # 2. Create clone prompt via Qwen3-TTS
    lang_name = iso_to_language(language)
    prompt_bytes = create_clone_prompt(
        ref_audio_path=str(ref_dest),
        ref_text=transcript,
        model_id=model_id,
    )
    logger.info("Created clone prompt (%d bytes) for voice %s", len(prompt_bytes), voice_id)

    # 3. Save clone prompt binary to NFS
    prompt_path = vdir / "clone_prompt.bin"
    prompt_path.write_bytes(prompt_bytes)

    # 4. Insert into database
    await execute_write(
        "INSERT INTO voices (id, name, created_by, source_type, source_url, "
        "language, ref_audio, transcript, model_id, is_public) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            voice_id,
            name,
            user_id,
            source_type,
            source_url,
            language,
            str(ref_dest),
            transcript,
            model_id,
            True,
        ),
    )
    logger.info("Inserted voice %s into database", voice_id)

    # 5. Fetch and return the created voice
    voice = await get_voice(voice_id)
    assert voice is not None, f"Voice {voice_id} was just inserted but not found"
    return voice


async def delete_voice(voice_id: str, user_id: str) -> bool:
    """Delete a cloned voice.

    Only the creator may delete a voice.  Returns ``True`` if the voice
    was found and deleted, ``False`` otherwise.
    """
    # Verify the caller is the creator
    voice = await get_voice(voice_id)
    if voice is None or voice.created_by != user_id:
        return False

    # Remove NFS directory
    vdir = VOICES_DIR / voice_id
    if vdir.exists():
        shutil.rmtree(str(vdir))
        logger.info("Removed voice directory %s", vdir)

    # Delete from database
    await execute_write("DELETE FROM voices WHERE id = ?", (voice_id,))
    logger.info("Deleted voice %s from database", voice_id)
    return True


def get_clone_prompt_bytes(voice_id: str) -> bytes | None:
    """Read the serialised clone prompt from NFS, or ``None`` if missing."""
    prompt_path = VOICES_DIR / voice_id / "clone_prompt.bin"
    if prompt_path.exists():
        return prompt_path.read_bytes()
    return None


def is_cloned_voice(voice_id: str) -> bool:
    """Check whether *voice_id* refers to a cloned voice (UUID) vs. a built-in voice."""
    try:
        uuid.UUID(voice_id)
        return True
    except ValueError:
        return False
