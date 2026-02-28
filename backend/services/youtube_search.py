"""YouTube search and audio download service using yt-dlp.

Used for voice cloning: search for a person's name on YouTube,
download a short audio clip of them speaking, which is later
transcribed by Whisper and used as a voice cloning reference.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_YOUTUBE_URL_RE = re.compile(
    r'^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}'
)


@dataclass
class YouTubeResult:
    """A single YouTube search result."""

    video_id: str
    title: str
    duration_seconds: int
    url: str
    thumbnail: str


async def search_youtube(
    query: str, max_results: int = 5
) -> list[YouTubeResult]:
    """Search YouTube for videos matching *query* using yt-dlp.

    Args:
        query: The search terms (e.g. a person's name).
        max_results: Maximum number of results to return.

    Returns:
        A list of :class:`YouTubeResult` instances.
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{max_results}:{query}",
    ]

    logger.info("Running YouTube search: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode().strip()
        logger.error("yt-dlp search failed (exit %d): %s", proc.returncode, error_msg)
        raise RuntimeError(f"yt-dlp search failed: {error_msg}")

    results: list[YouTubeResult] = []
    for line in stdout.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON line from yt-dlp output")
            continue

        video_id = data.get("id", "")
        title = data.get("title", "")
        duration = int(data.get("duration") or 0)
        url = data.get("url") or data.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
        thumbnail = data.get("thumbnail") or data.get("thumbnails", [{}])[-1].get("url", "")

        results.append(
            YouTubeResult(
                video_id=video_id,
                title=title,
                duration_seconds=duration,
                url=url,
                thumbnail=thumbnail,
            )
        )

    logger.info("YouTube search returned %d results for query %r", len(results), query)
    return results


async def download_audio(
    video_url: str,
    output_dir: Path,
    max_duration_seconds: int = 30,
) -> Path:
    """Download audio from a YouTube video and trim it.

    Downloads the best-quality audio as WAV, then trims to
    *max_duration_seconds* so it is suitable as a short voice
    cloning reference clip.

    Args:
        video_url: Full YouTube video URL.
        output_dir: Directory where the WAV file will be saved.
        max_duration_seconds: Maximum length of the output clip in seconds.

    Returns:
        Path to the trimmed WAV file.

    Raises:
        RuntimeError: If yt-dlp exits with a non-zero code.
        ValueError: If the URL is not a valid YouTube URL.
    """
    if not _YOUTUBE_URL_RE.match(video_url):
        raise ValueError("Only YouTube URLs are accepted")

    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp template for output filename (without extension -- yt-dlp adds it)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", output_template,
        "--no-playlist",
        video_url,
    ]

    logger.info("Downloading audio: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode().strip()
        logger.error("yt-dlp download failed (exit %d): %s", proc.returncode, error_msg)
        raise RuntimeError(f"yt-dlp download failed: {error_msg}")

    # Find the downloaded WAV file (yt-dlp fills in the id and ext)
    wav_files = sorted(output_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    if not wav_files:
        raise RuntimeError("yt-dlp did not produce a WAV file")

    downloaded_path = wav_files[0]
    logger.info("Downloaded audio to %s", downloaded_path)

    # Trim to max_duration_seconds using pydub (lazy import -- not available outside Docker)
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(str(downloaded_path))
    max_ms = max_duration_seconds * 1000

    if len(audio) > max_ms:
        trimmed = audio[:max_ms]
        trimmed_path = downloaded_path.with_stem(downloaded_path.stem + "_trimmed")
        trimmed.export(str(trimmed_path), format="wav")
        logger.info(
            "Trimmed audio from %d ms to %d ms -> %s",
            len(audio),
            max_ms,
            trimmed_path,
        )
        # Clean up the untrimmed original
        downloaded_path.unlink()
        return trimmed_path

    # Already short enough -- no trimming needed
    return downloaded_path
