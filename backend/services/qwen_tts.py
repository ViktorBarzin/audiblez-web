"""Qwen3-TTS voice cloning service.

Wraps the qwen-tts package for lazy model loading, voice clone prompt
creation, speech generation, and chapter-level audio production.
The model is loaded once and kept in GPU memory between jobs.
"""

import io
import os
import pickle
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QWEN_TTS_MODEL: str = os.environ.get(
    "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

AVAILABLE_MODELS: dict[str, dict] = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {"name": "1.7B Base", "vram_gb": 5},
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {"name": "0.6B Base", "vram_gb": 2},
}

LANGUAGE_MAP: dict[str, str] = {
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

# Sample texts used by generate_sample() for each supported language.
_SAMPLE_TEXTS: dict[str, str] = {
    "English": "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Chinese": "快速的棕色狐狸跳过了河边懒惰的狗。",
    "Japanese": "素早い茶色の狐が怠惰な犬を飛び越えました。",
    "Korean": "빠른 갈색 여우가 게으른 개를 뛰어넘었습니다.",
    "German": "Der schnelle braune Fuchs springt ueber den faulen Hund am Flussufer.",
    "French": "Le rapide renard brun saute par-dessus le chien paresseux pres de la riviere.",
    "Russian": "Быстрая бурая лиса перепрыгнула через ленивую собаку у реки.",
    "Portuguese": "A raposa marrom rapida salta sobre o cachorro preguicoso perto do rio.",
    "Spanish": "El rapido zorro marron salta sobre el perro perezoso cerca del rio.",
    "Italian": "La veloce volpe marrone salta sopra il cane pigro vicino al fiume.",
}

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_model = None
_loaded_model_id: str | None = None


def _get_model(model_id: str | None = None):
    """Load the Qwen3-TTS model lazily.

    If *model_id* differs from the currently loaded model the old model is
    deleted and CUDA memory is freed before loading the new one.
    """
    global _model, _loaded_model_id

    from qwen_tts import Qwen3TTSModel  # deferred to avoid import-time errors

    target_model = model_id or QWEN_TTS_MODEL

    if _model is not None and _loaded_model_id == target_model:
        return _model

    # Release previous model if switching
    if _model is not None:
        del _model
        _model = None
        _loaded_model_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _model = Qwen3TTSModel.from_pretrained(
        target_model, device_map=device, dtype=dtype
    )
    _loaded_model_id = target_model
    return _model


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def create_clone_prompt(
    ref_audio_path: str | Path,
    ref_text: str,
    model_id: str | None = None,
) -> bytes:
    """Create a voice-clone prompt from a reference audio clip and its text.

    Returns the prompt serialised as *pickle* bytes so it can be stored in a
    database or passed between processes.
    """
    model = _get_model(model_id)
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path), ref_text=ref_text
    )
    return pickle.dumps(prompt)


def generate_speech(
    text: str,
    clone_prompt_bytes: bytes,
    language: str = "English",
    model_id: str | None = None,
) -> tuple[np.ndarray, int]:
    """Generate speech audio from *text* using a previously created clone prompt.

    Returns a tuple of ``(waveform_array, sample_rate)``.
    """
    model = _get_model(model_id)
    prompt = pickle.loads(clone_prompt_bytes)
    wavs, sr = model.generate_voice_clone(
        text=text, language=language, voice_clone_prompt=prompt
    )
    return wavs[0], sr


def generate_chapter_audio(
    chapter_text: str,
    clone_prompt_bytes: bytes,
    output_path: str | Path,
    language: str = "English",
    model_id: str | None = None,
    speed: float = 1.0,
    max_segment_words: int = 500,
) -> Path:
    """Synthesise a full chapter and write the result to *output_path*.

    Long chapters are split into segments of up to *max_segment_words* words,
    generated individually, and concatenated.  If *speed* is not 1.0 the
    final audio is time-stretched via ``pydub`` frame-rate manipulation.

    Returns the ``Path`` to the written audio file.
    """
    output_path = Path(output_path)
    segments = _split_text(chapter_text, max_words=max_segment_words)

    all_audio: list[np.ndarray] = []
    sample_rate: int = 24000  # fallback; overwritten on first segment

    for segment in segments:
        wav, sr = generate_speech(
            text=segment,
            clone_prompt_bytes=clone_prompt_bytes,
            language=language,
            model_id=model_id,
        )
        sample_rate = sr
        all_audio.append(wav)

    combined = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)

    # Apply speed adjustment if needed
    if speed != 1.0 and len(combined) > 0:
        from pydub import AudioSegment

        # Convert numpy float array to 16-bit PCM bytes for pydub
        pcm_16 = (combined * 32767).astype(np.int16)
        audio_seg = AudioSegment(
            data=pcm_16.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        # Manipulate frame rate to change speed without pitch correction
        adjusted = audio_seg._spawn(
            audio_seg.raw_data,
            overrides={"frame_rate": int(audio_seg.frame_rate * speed)},
        )
        # Restore original frame rate so the file plays at the right clock
        adjusted = adjusted.set_frame_rate(sample_rate)

        # Convert back to numpy
        samples = np.array(adjusted.get_array_of_samples(), dtype=np.float32)
        combined = samples / 32767.0

    sf.write(str(output_path), combined, sample_rate)
    return output_path


def generate_sample(
    clone_prompt_bytes: bytes,
    language: str = "English",
    model_id: str | None = None,
) -> bytes:
    """Generate a short preview clip and return it as WAV bytes."""
    sample_text = _SAMPLE_TEXTS.get(language, _SAMPLE_TEXTS["English"])

    wav, sr = generate_speech(
        text=sample_text,
        clone_prompt_bytes=clone_prompt_bytes,
        language=language,
        model_id=model_id,
    )

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_available_models() -> list[dict]:
    """Return a list of available Qwen3-TTS model descriptors."""
    return [
        {"id": model_id, "name": info["name"], "vram_gb": info["vram_gb"]}
        for model_id, info in AVAILABLE_MODELS.items()
    ]


def iso_to_language(iso_code: str) -> str:
    """Convert an ISO 639-1 language code to the Qwen3-TTS language name.

    Falls back to ``"English"`` for unknown codes.
    """
    return LANGUAGE_MAP.get(iso_code, "English")


def _split_text(text: str, max_words: int = 500) -> list[str]:
    """Split *text* into segments of at most *max_words* words.

    Splits preferentially at sentence boundaries (``.``, ``!``, ``?``) so
    that segments end on natural pauses.
    """
    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    segments: list[str] = []
    current: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        if current_word_count + sentence_word_count > max_words and current:
            segments.append(" ".join(current))
            current = []
            current_word_count = 0

        # Handle a single sentence longer than max_words
        if sentence_word_count > max_words:
            # Flush anything accumulated so far
            if current:
                segments.append(" ".join(current))
                current = []
                current_word_count = 0

            # Chunk the oversized sentence by word count
            words = sentence_words
            for i in range(0, len(words), max_words):
                segments.append(" ".join(words[i : i + max_words]))
        else:
            current.append(sentence)
            current_word_count += sentence_word_count

    if current:
        segments.append(" ".join(current))

    return segments
