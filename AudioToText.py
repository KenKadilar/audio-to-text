"""
AudioToText - PyQt6 GUI for multi-track audio transcription.

Takes a multi-track audio/video file (e.g. an OBS-recorded .mkv where each
speaker has their own dedicated track plus a combined mix), splits each
selected track into ~9-minute WAV chunks, sends them to OpenAI for
transcription (diarization for the combined track, prompt-supported
transcription for the solos), and assembles a chronological speaker-labeled
timeline.

The interesting bit is the per-chunk speaker resolution step. OpenAI's
diarize model assigns A/B/C labels independently per audio chunk - "A" in
chunk 1 may correspond to a different person than "A" in chunk 2. A single
global remap therefore mis-attributes turns at every chunk boundary. The
fix is to resolve labels per chunk by comparing each raw label's text
against the solo reference tracks for the same chunk, using a containment-
via-smaller similarity metric (more robust than symmetric Jaccard when one
solo is sparse for that chunk).

Expected portable layout:

    <app dir>/
        AudioToText.py
        requirements.txt
        AudioToText.ico              # optional icon
        DATA/                        # all runtime state lives here (gitignored)
            TOOLS/
                ffmpeg.exe
                ffprobe.exe
            AudioToText/
                .env                 # contains OPENAI_API_KEY=...
                Input/               # source recordings
                Extracts/<session>/  # per-session output folders

Outputs (per session, under transcripts/):

    speaker_a.md                 Solo Speaker A transcript
    speaker_b.md                 Solo Speaker B transcript
    combined.md                  Chronological Speaker A/B timeline, resolved
                                 per chunk against the solo references
    cleanup_prompt.md            Prompt for a downstream wording-polish AI
    ai_cleanup_bundle.md         Cleanup prompt + all three above, concatenated

A "Reprocess From Raw JSON" button regenerates these outputs from existing
raw_json/*_segments.json files without re-calling OpenAI.

Workflow:
    1. Browse a recording (.mkv/.mp3/.mp4/.wav/etc.).
    2. Scan audio tracks with ffprobe.
    3. Assign roles to each track:
         - Combined / Mixed Track: both/all speakers, typically diarized.
         - Speaker A (Solo): clean solo track for participant A.
         - Speaker B (Solo): clean solo track for participant B.
         - Ignore: skip.
    4. Extract selected tracks into 16 kHz mono WAV chunks.
    5. Optional fallback: synthesize a combined mix from the solos if no
       real mixed track exists.
    6. Transcribe the combined track with diarization for chronology.
    7. Transcribe the solo tracks with prompt-supported transcription for
       cleaner wording.
    8. Write speaker_a.md, speaker_b.md, combined.md, ai_cleanup_bundle.md.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "AudioToText"
APP_VERSION = "3.1"
APP_ICON_FILE = "AudioToText.ico"

SUPPORTED_INPUT_EXTS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".webm",
    ".mp4",
    ".mkv",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
}

ROLE_IGNORE = "Ignore"
ROLE_MAIN = "Combined / Mixed Track"
ROLE_SPEAKER_B = "Speaker B (Solo)"
ROLE_SPEAKER_A = "Speaker A (Solo)"
ROLE_REFERENCE = "Other Reference Track"
ROLE_OTHER_MAIN = "Other Combined / Mixed Track"

ROLE_OPTIONS = [
    ROLE_IGNORE,
    ROLE_MAIN,
    ROLE_SPEAKER_A,
    ROLE_SPEAKER_B,
    ROLE_REFERENCE,
    ROLE_OTHER_MAIN,
]

MAIN_ROLES = {ROLE_MAIN, ROLE_OTHER_MAIN}
REFERENCE_ROLES = {ROLE_SPEAKER_A, ROLE_SPEAKER_B, ROLE_REFERENCE}

SOLO_TAG_BY_ROLE = {
    ROLE_SPEAKER_A: "Speaker A",
    ROLE_SPEAKER_B: "Speaker B",
}

DEFAULT_PROMPT = (
    "Spoken-conversation transcript. Preserve the original language as spoken; "
    "do not translate, do not rewrite into formal prose, do not summarize. "
    "Add punctuation for readability. Preserve names, dates, technical terms, "
    "product / app names, and the speaker's exact wording as accurately as possible."
)

CLEANUP_PROMPT_TEMPLATE = """# Cleanup Prompt for AI

You are given three transcript files from the same recording.

1. SPEAKER A - Solo audio track of one participant. Single speaker throughout. All lines tagged "Speaker A".
2. SPEAKER B - Solo audio track of the other participant. Single speaker throughout. All lines tagged "Speaker B".
3. COMBINED - Full chronological conversation, already labeled "Speaker A" or "Speaker B". The labels were resolved per chunk by comparing the diarized combined-track output against the two solo references. Backchannel noise has been dropped and consecutive same-speaker segments merged into turns.

Your job: polish COMBINED into a final readable transcript by fixing transcription artifacts.

Common issues to fix:
- **Hallucinated loops**: speech-to-text models sometimes repeat one sentence dozens of times when audio goes silent. Drop them.
- **Phonetic spelling errors**: names, app names, technical terms, domain-specific words may be misspelled. Cross-check against the solo transcripts and use the clearer spelling.
- **Mid-sentence speaker splits**: if a phrase starts attributed to one speaker and a fragment is attributed to the other, but the whole sentence makes sense as one speaker's, fix the attribution.
- **Wrong speaker attribution**: the per-chunk solo cross-reference is good but not perfect. If a turn's content clearly belongs to the other speaker, fix the label.

Do NOT:
- Translate. Preserve the original language(s) as spoken.
- Summarize or rewrite into formal prose.
- Drop substantive content.
- Change timestamps.

Output format (same as COMBINED, with corrections applied):
[00:00:00.000 - 00:00:05.000] Speaker A: ...
[00:00:05.200 - 00:00:09.000] Speaker B: ...
"""


# ─────────────────────────────────────────────────────────────────────────────
# Portable paths / resources
# ─────────────────────────────────────────────────────────────────────────────

def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_path(filename: str) -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / filename
    return get_app_dir() / filename


def apply_app_icon(app_or_window: Any) -> None:
    icon_path = get_resource_path(APP_ICON_FILE)
    if icon_path.exists():
        app_or_window.setWindowIcon(QIcon(str(icon_path)))


def data_root() -> Path:
    return get_app_dir() / "DATA" / "AudioToText"


def extracts_root() -> Path:
    return data_root() / "Extracts"


def tools_dir() -> Path:
    return get_app_dir() / "DATA" / "TOOLS"


def find_tool(tool_name: str) -> str:
    """Prefer bundled DATA/TOOLS executable, then PATH."""
    exe_name = tool_name + (".exe" if os.name == "nt" else "")
    bundled = tools_dir() / exe_name
    if bundled.exists():
        return str(bundled)

    found = shutil.which(tool_name) or shutil.which(exe_name)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find {exe_name}. Put it in DATA/TOOLS or install it on PATH."
    )


def ensure_base_dirs() -> None:
    data_root().mkdir(parents=True, exist_ok=True)
    extracts_root().mkdir(parents=True, exist_ok=True)
    tools_dir().mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_name(value: str, fallback: str = "item") -> str:
    value = value.strip()
    value = re.sub(r"[<>:\"/\\|?*]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._ ")
    return value or fallback


def slugify(value: str, fallback: str = "track") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def format_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    total_seconds, ms = divmod(total_ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_plain_data(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_plain_data(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_plain_data(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain_data(v) for k, v in obj.items()}

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass

    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {k: to_plain_data(v) for k, v in data.items() if not k.startswith("_")}

    return str(obj)


def extract_text(transcript: Any) -> str:
    text = get_attr_or_key(transcript, "text", "")
    if text:
        return str(text).strip()

    plain = to_plain_data(transcript)
    if isinstance(plain, dict):
        text = plain.get("text", "")
        if text:
            return str(text).strip()

        segments = plain.get("segments") or []
        parts = [str(seg.get("text", "")).strip() for seg in segments if seg.get("text")]
        return " ".join(part for part in parts if part).strip()

    return ""


def extract_segments(transcript: Any) -> list[Any]:
    segments = get_attr_or_key(transcript, "segments", [])
    if segments:
        return segments

    plain = to_plain_data(transcript)
    if isinstance(plain, dict):
        return plain.get("segments", []) or []

    return []


def get_duration_seconds(path: Path, ffprobe_path: str) -> float:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ffprobe / ffmpeg
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioTrack:
    audio_position: int      # 0-based audio stream number for ffmpeg 0:a:N mapping
    stream_index: int        # container stream index
    map_spec: str            # e.g. 0:a:2
    codec: str
    channels: str
    sample_rate: str
    title: str
    language: str

    def display_name(self) -> str:
        bits = [f"Track {self.audio_position + 1}", self.map_spec]
        if self.title:
            bits.append(self.title)
        if self.language:
            bits.append(self.language)
        return " - ".join(bits)


def scan_audio_tracks(input_path: Path) -> list[AudioTrack]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
        raise ValueError(
            f"Unsupported file extension: {input_path.suffix}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_INPUT_EXTS))}"
        )

    ffprobe_path = find_tool("ffprobe")
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate:stream_tags=title,language",
        "-of",
        "json",
        str(input_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", []) or []

    tracks: list[AudioTrack] = []
    for pos, stream in enumerate(streams):
        tags = stream.get("tags") or {}
        tracks.append(
            AudioTrack(
                audio_position=pos,
                stream_index=int(stream.get("index", pos)),
                map_spec=f"0:a:{pos}",
                codec=str(stream.get("codec_name", "")),
                channels=str(stream.get("channels", "")),
                sample_rate=str(stream.get("sample_rate", "")),
                title=str(tags.get("title", "")),
                language=str(tags.get("language", "")),
            )
        )

    return tracks


def build_session_dir(input_path: Path) -> Path:
    base = sanitize_name(input_path.stem, "session")
    return extracts_root() / base


def chunk_dir_for_track(session_dir: Path, track_slug: str) -> Path:
    return session_dir / "extracted_audio" / track_slug / "chunks"


def ffmpeg_common_chunk_output_args(chunk_seconds: int, chunk_pattern: str) -> list[str]:
    return [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        chunk_pattern,
    ]


def clear_chunk_folder(chunks_dir: Path) -> None:
    for old in chunks_dir.glob("chunk_*.wav"):
        try:
            old.unlink()
        except OSError:
            pass


def extract_track_to_chunks(
    input_path: Path,
    track: dict[str, Any],
    session_dir: Path,
    chunk_seconds: int,
    clear_old_chunks: bool,
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    ffmpeg_path = find_tool("ffmpeg")
    label = str(track.get("label") or f"Track {track.get('audio_position', 0) + 1}").strip()
    role = str(track.get("role") or ROLE_REFERENCE).strip()
    audio_position = int(track["audio_position"])
    track_slug = slugify(f"{role}_{label}_track_{audio_position + 1}", f"track_{audio_position + 1}")

    chunks_dir = chunk_dir_for_track(session_dir, track_slug)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if clear_old_chunks:
        clear_chunk_folder(chunks_dir)

    chunk_pattern = str(chunks_dir / "chunk_%03d.wav")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:a:{audio_position}",
        "-vn",
        *ffmpeg_common_chunk_output_args(chunk_seconds, chunk_pattern),
    ]

    log_fn(f"Extracting {label} [{role}] from {track.get('map_spec')} into WAV chunks...")
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while extracting {label}.\n\n"
            f"COMMAND:\n{' '.join(cmd)}\n\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )

    chunk_files = sorted(Path(p) for p in glob.glob(str(chunks_dir / "chunk_*.wav")))
    if not chunk_files:
        raise RuntimeError(f"No chunks were created for {label}.")

    log_fn(f"Created {len(chunk_files)} chunks for {label}.")

    return {
        **track,
        "label": label,
        "role": role,
        "track_slug": track_slug,
        "chunks_dir": str(chunks_dir),
        "chunk_files": [str(p) for p in chunk_files],
        "generated_mix": False,
    }


def create_fallback_mix_chunks(
    input_path: Path,
    source_tracks: list[dict[str, Any]],
    session_dir: Path,
    chunk_seconds: int,
    clear_old_chunks: bool,
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Create a generated main ordered mix from selected reference tracks.

    This is meant as a fallback if the recording does not already contain a real
    mixed conversation track.
    """
    if len(source_tracks) < 2:
        raise ValueError("Fallback mix needs at least two selected reference tracks.")

    ffmpeg_path = find_tool("ffmpeg")
    track_slug = "generated_main_mix_from_references"
    chunks_dir = chunk_dir_for_track(session_dir, track_slug)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if clear_old_chunks:
        clear_chunk_folder(chunks_dir)

    stream_inputs = "".join(f"[0:a:{int(track['audio_position'])}]" for track in source_tracks)
    filter_complex = (
        f"{stream_inputs}"
        f"amix=inputs={len(source_tracks)}:duration=longest:dropout_transition=0:normalize=1,"
        "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[mix]"
    )

    chunk_pattern = str(chunks_dir / "chunk_%03d.wav")
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[mix]",
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        chunk_pattern,
    ]

    labels = ", ".join(str(track.get("label") or track.get("map_spec")) for track in source_tracks)
    log_fn(f"Creating fallback main ordered mix from: {labels}")
    completed = subprocess.run(cmd, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while creating fallback mixed track.\n\n"
            f"COMMAND:\n{' '.join(cmd)}\n\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )

    chunk_files = sorted(Path(p) for p in glob.glob(str(chunks_dir / "chunk_*.wav")))
    if not chunk_files:
        raise RuntimeError("No chunks were created for fallback mixed track.")

    log_fn(f"Created {len(chunk_files)} chunks for generated main mix.")

    return {
        "audio_position": -1,
        "stream_index": -1,
        "map_spec": "generated_mix",
        "codec": "pcm_s16le",
        "channels": "1",
        "sample_rate": "16000",
        "title": "Generated mix from selected reference tracks",
        "language": "",
        "label": "Generated Main Mix",
        "role": ROLE_MAIN,
        "track_slug": track_slug,
        "chunks_dir": str(chunks_dir),
        "chunk_files": [str(p) for p in chunk_files],
        "generated_mix": True,
        "source_tracks": source_tracks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Speaker labeling
# ─────────────────────────────────────────────────────────────────────────────

def resolve_speaker_tag(role: str, raw_speaker: str, fallback_label: str) -> str:
    """Pick the speaker tag for a segment based on role.

    - ROLE_SPEAKER_A always becomes "Speaker A".
    - ROLE_SPEAKER_B always becomes "Speaker B".
    - MAIN roles keep the raw diarization label (remapped later to S1, S2, ...).
    - Anything else falls back to the user-provided label.
    """
    if role in SOLO_TAG_BY_ROLE:
        return SOLO_TAG_BY_ROLE[role]
    if role in MAIN_ROLES:
        return (raw_speaker or fallback_label or "S?").strip()
    return fallback_label or "Speaker"


def remap_main_speakers(segments: list[dict[str, Any]]) -> dict[str, str]:
    """Renumber diarization speaker labels (A/B/C...) to S1/S2/S3... in order of first appearance.

    Mutates segments in place. Returns the mapping so callers can log it.
    This is a fallback used only when no solo Speaker A / Speaker B tracks
    are present; when solos exist, prefer resolve_combined_speakers() instead.
    """
    mapping: dict[str, str] = {}
    counter = 0
    for seg in segments:
        raw = str(seg.get("speaker", "")).strip()
        if not raw:
            continue
        if raw not in mapping:
            counter += 1
            mapping[raw] = f"S{counter}"
    for seg in segments:
        raw = str(seg.get("speaker", "")).strip()
        if raw and raw in mapping:
            seg["raw_speaker"] = raw
            seg["speaker"] = mapping[raw]
    return mapping


_BACKCHANNEL_RE = re.compile(
    r"^(mm+|hmm+|mhm+|mm-?hmm|hı+\s*hı+|hi+\s*hi+|hi+\s*hı+|hh+|h-h|huh+|mh-hm|h|m|ah+|eh+|oh+|öh+|uh+)$",
    re.IGNORECASE,
)


def is_backchannel(text: str) -> bool:
    """True if text is just a nonverbal acknowledgment (Mm, Hmm, Mhm, etc).

    Substantive single-word responses ('yes', 'no', 'okay', etc.) are NOT
    treated as backchannels and are preserved.
    """
    clean = text.strip().strip(".,!?…").strip()
    if not clean:
        return True
    return bool(_BACKCHANNEL_RE.match(clean))


def _tokenize_for_match(text: str) -> set[str]:
    """Tokenize for content-similarity matching: lowercase, drop punctuation, drop 1-char tokens."""
    lowered = text.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return {token for token in no_punct.split() if len(token) >= 2}


def _containment(label_tokens: set[str], solo_tokens: set[str]) -> float:
    """|label ∩ solo| / min(|label|, |solo|).

    This is robust to asymmetric solo sizes. Symmetric Jaccard is misleading
    when one solo is sparse for a chunk (e.g. one participant is silent while
    the other speaks at length): the larger solo accumulates more spurious
    matches via common filler words and "wins" even when the label is
    actually the smaller solo's speaker. Containment-via-smaller normalises
    by the smaller set, so a label that contains most of a sparse solo's
    vocabulary is correctly identified.
    """
    if not label_tokens or not solo_tokens:
        return 0.0
    return len(label_tokens & solo_tokens) / min(len(label_tokens), len(solo_tokens))


# Below this a label is a one-word turn where overlap with either solo is noise.
_MIN_LABEL_TOKENS_FOR_RESOLUTION = 5
# Winner must beat the loser by this multiple, else the raw S-label stays.
_MIN_RESOLUTION_MARGIN = 1.3


def resolve_combined_speakers(
    main_segments: list[dict[str, Any]],
    speaker_a_segments: list[dict[str, Any]],
    speaker_b_segments: list[dict[str, Any]],
    log_fn: Callable[[str], None],
) -> tuple[dict[int, dict[str, str]], list[str]]:
    """Resolve diarization labels in main_segments to "Speaker A" or "Speaker B"
    per chunk.

    Why per chunk: OpenAI's diarize model assigns labels independently for each
    audio chunk, starting fresh at "A" each time. So a single global remap is
    wrong; "A" in chunk 1 may be one participant while "A" in chunk 2 is the
    other. This function groups main segments by chunk_index, concatenates the
    text for each raw label within that chunk, and compares it against the
    Speaker A and Speaker B solo text from the same chunk using
    containment-via-smaller.

    Mutates main_segments in place: sets seg["raw_speaker"] to the original
    label and seg["speaker"] to "Speaker A" or "Speaker B" where confidence is
    high enough. Labels with insufficient confidence are left with raw S-labels
    (renumbered to S1/S2/S3 across the whole session via a final fallback pass)
    so the downstream cleanup AI can resolve them by reading context.

    Returns: ({chunk_index: {raw_label: "Speaker A" | "Speaker B"}},
              [unresolved chunk-label notes])
    """
    if not main_segments:
        return {}, []

    main_by_chunk: dict[int, list[dict[str, Any]]] = {}
    for seg in main_segments:
        ci = int(seg.get("chunk_index", 0) or 0)
        main_by_chunk.setdefault(ci, []).append(seg)

    def solo_tokens_by_chunk(segments: list[dict[str, Any]]) -> dict[int, set[str]]:
        bucket: dict[int, list[str]] = {}
        for seg in segments:
            ci = int(seg.get("chunk_index", 0) or 0)
            bucket.setdefault(ci, []).append(str(seg.get("text", "")).strip())
        return {ci: _tokenize_for_match(" ".join(parts)) for ci, parts in bucket.items()}

    tokens_a_by_chunk = solo_tokens_by_chunk(speaker_a_segments)
    tokens_b_by_chunk = solo_tokens_by_chunk(speaker_b_segments)

    resolution: dict[int, dict[str, str]] = {}
    unresolved_notes: list[str] = []

    for ci, segs in sorted(main_by_chunk.items()):
        tokens_a = tokens_a_by_chunk.get(ci, set())
        tokens_b = tokens_b_by_chunk.get(ci, set())

        text_by_label: dict[str, list[str]] = {}
        for seg in segs:
            raw = (seg.get("raw_speaker") or seg.get("speaker") or "").strip()
            if not raw:
                continue
            text_by_label.setdefault(raw, []).append(str(seg.get("text", "")).strip())

        label_to_speaker: dict[str, str] = {}
        unresolved_in_chunk: list[str] = []
        for raw, texts in text_by_label.items():
            label_tokens = _tokenize_for_match(" ".join(texts))
            if len(label_tokens) < _MIN_LABEL_TOKENS_FOR_RESOLUTION:
                unresolved_in_chunk.append(f"{raw}(<{_MIN_LABEL_TOKENS_FOR_RESOLUTION}tok)")
                continue

            score_a = _containment(label_tokens, tokens_a)
            score_b = _containment(label_tokens, tokens_b)

            if score_a == 0.0 and score_b == 0.0:
                unresolved_in_chunk.append(f"{raw}(no-solo-overlap)")
                continue

            if score_a >= score_b * _MIN_RESOLUTION_MARGIN:
                label_to_speaker[raw] = "Speaker A"
            elif score_b >= score_a * _MIN_RESOLUTION_MARGIN:
                label_to_speaker[raw] = "Speaker B"
            else:
                unresolved_in_chunk.append(
                    f"{raw}(A={score_a:.2f},B={score_b:.2f})"
                )

        if label_to_speaker:
            resolution[ci] = label_to_speaker
            log_fn(f"  Chunk {ci}: {label_to_speaker}")
        if unresolved_in_chunk:
            note = f"  Chunk {ci} unresolved labels: {', '.join(unresolved_in_chunk)}"
            log_fn(note)
            unresolved_notes.append(note.strip())

    # Apply resolved labels; unresolved ones get a global S1/S2/S3 remap in first-appearance order.
    unresolved_remap: dict[str, str] = {}
    counter = 0
    for seg in main_segments:
        ci = int(seg.get("chunk_index", 0) or 0)
        raw = (seg.get("raw_speaker") or seg.get("speaker") or "").strip()
        resolved = resolution.get(ci, {}).get(raw)
        if resolved:
            seg["raw_speaker"] = raw
            seg["speaker"] = resolved
        elif raw:
            # Stable S-label keeps these distinct from the chunk-local diarizer letters.
            key = f"chunk{ci}:{raw}"
            if key not in unresolved_remap:
                counter += 1
                unresolved_remap[key] = f"S{counter}"
            seg["raw_speaker"] = raw
            seg["speaker"] = unresolved_remap[key]

    return resolution, unresolved_notes


def drop_backchannel_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out segments whose text is just a nonverbal acknowledgment."""
    return [seg for seg in segments if not is_backchannel(str(seg.get("text", "")))]


def merge_consecutive_same_speaker(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive segments with the same speaker into one turn each.

    Returns a new list of new dicts. Does not mutate input.
    """
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and merged[-1].get("speaker") == seg.get("speaker"):
            prev = merged[-1]
            prev["end"] = max(float(prev.get("end", 0.0) or 0.0), float(seg.get("end", 0.0) or 0.0))
            prev_text = str(prev.get("text", "")).strip()
            new_text = str(seg.get("text", "")).strip()
            if new_text:
                prev["text"] = (prev_text + " " + new_text).strip() if prev_text else new_text
        else:
            merged.append(dict(seg))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI transcription
# ─────────────────────────────────────────────────────────────────────────────

def load_openai_client() -> OpenAI:
    env_path = data_root() / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY was not found. Expected .env here:\n{env_path}"
        )
    return OpenAI(api_key=api_key)


def transcribe_chunk(
    client: OpenAI,
    chunk_path: Path,
    model: str,
    prompt: str,
) -> Any:
    with open(chunk_path, "rb") as audio_file:
        kwargs: dict[str, Any] = {
            "model": model,
            "file": audio_file,
        }

        if model == "gpt-4o-transcribe-diarize":
            # Gives segments with start/end/speaker, but does not support prompt.
            kwargs["response_format"] = "diarized_json"
            kwargs["chunking_strategy"] = "auto"
        elif model == "whisper-1":
            kwargs["response_format"] = "verbose_json"
            if prompt.strip():
                kwargs["prompt"] = prompt.strip()
        else:
            # gpt-4o-transcribe / gpt-4o-mini-transcribe support json, not verbose_json.
            kwargs["response_format"] = "json"
            if prompt.strip():
                kwargs["prompt"] = prompt.strip()

        return client.audio.transcriptions.create(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Output rendering
# ─────────────────────────────────────────────────────────────────────────────

def format_segments_markdown(
    title: str,
    segments: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> str:
    lines = [f"# {title}", "", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for note in notes or []:
        lines.append(f"> {note}")
    if notes:
        lines.append("")

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = format_time(float(seg.get("start", 0.0)))
        end = format_time(float(seg.get("end", 0.0)))
        speaker = str(seg.get("speaker", "Speaker"))
        lines.append(f"[{start} - {end}] **{speaker}:** {text}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_cleanup_bundle(
    bundle_path: Path,
    sections: list[tuple[str, Path | None]],
) -> None:
    parts = [CLEANUP_PROMPT_TEMPLATE.strip(), "", "---", ""]
    for title, file_path in sections:
        if not file_path or not file_path.exists():
            continue
        body = read_text_if_exists(file_path).strip()
        if not body:
            continue
        parts.extend([f"# {title}", "", body, "", "---", ""])
    bundle_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")


def build_session_outputs(
    session_dir: Path,
    all_segments: list[dict[str, Any]],
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Group segments by role, resolve combined-track speakers per chunk against
    the Speaker A / Speaker B solo references, and write the four output files.

    Assumes:
      - Segments tagged with ROLE_SPEAKER_A or ROLE_SPEAKER_B already have
        role-forced speaker labels (set by resolve_speaker_tag during
        transcribe / reprocess).
      - Segments tagged with a MAIN role still hold raw diarization labels.
    Mutates main segments in place (speaker -> "Speaker A" / "Speaker B" / S* fallback).
    """
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    speaker_a_segments = [seg for seg in all_segments if seg.get("role") == ROLE_SPEAKER_A]
    speaker_b_segments = [seg for seg in all_segments if seg.get("role") == ROLE_SPEAKER_B]
    main_segments = [seg for seg in all_segments if seg.get("role") in MAIN_ROLES]
    extra_segments = [seg for seg in all_segments if seg.get("role") == ROLE_REFERENCE]

    for seg in speaker_a_segments:
        seg["speaker"] = "Speaker A"
    for seg in speaker_b_segments:
        seg["speaker"] = "Speaker B"

    resolution_notes: list[str] = []
    if main_segments:
        if speaker_a_segments and speaker_b_segments:
            log_fn("Resolving combined-track diarization against solo references...")
            resolution, unresolved = resolve_combined_speakers(
                main_segments, speaker_a_segments, speaker_b_segments, log_fn
            )
            if resolution:
                resolution_notes.append(
                    "Combined-track speakers were resolved per chunk by content matching "
                    "(containment-via-smaller, margin 1.3x) against the Speaker A and "
                    f"Speaker B solo transcripts: {dict(resolution)}"
                )
            else:
                resolution_notes.append(
                    "Combined-track diarization labels could not be matched to the solo references; raw labels preserved."
                )
            if unresolved:
                resolution_notes.append(
                    "Some labels were ambiguous and left as S-labels for cleanup-AI review: "
                    + "; ".join(n.strip() for n in unresolved)
                )
        else:
            mapping = remap_main_speakers(main_segments)
            if mapping:
                resolution_notes.append(
                    "No Speaker A / Speaker B solo references available; combined-track "
                    f"diarization labels were renumbered to S1/S2/S3 in first-appearance order: {mapping}"
                )
                log_fn(resolution_notes[-1])

    speaker_a_segments.sort(key=lambda item: float(item.get("start", 0.0)))
    speaker_b_segments.sort(key=lambda item: float(item.get("start", 0.0)))
    main_segments.sort(key=lambda item: float(item.get("start", 0.0)))
    extra_segments.sort(key=lambda item: (str(item.get("source_label", "")), float(item.get("start", 0.0))))

    speaker_a_path = transcripts_dir / "speaker_a.md"
    speaker_b_path = transcripts_dir / "speaker_b.md"
    combined_path = transcripts_dir / "combined.md"
    extras_path = transcripts_dir / "extras.md"
    cleanup_prompt_path = transcripts_dir / "cleanup_prompt.md"
    bundle_path = transcripts_dir / "ai_cleanup_bundle.md"

    written: dict[str, str] = {}

    if speaker_a_segments:
        speaker_a_md = format_segments_markdown("Speaker A (Solo)", speaker_a_segments)
        speaker_a_path.write_text(speaker_a_md, encoding="utf-8")
        written["speaker_a_md"] = str(speaker_a_path)
        log_fn(f"Speaker A transcript: {speaker_a_path}")
    else:
        log_fn("No Speaker A-role segments found; skipping speaker_a.md.")

    if speaker_b_segments:
        speaker_b_md = format_segments_markdown("Speaker B (Solo)", speaker_b_segments)
        speaker_b_path.write_text(speaker_b_md, encoding="utf-8")
        written["speaker_b_md"] = str(speaker_b_path)
        log_fn(f"Speaker B transcript: {speaker_b_path}")
    else:
        log_fn("No Speaker B-role segments found; skipping speaker_b.md.")

    if main_segments:
        cleaned = drop_backchannel_segments(main_segments)
        merged = merge_consecutive_same_speaker(cleaned)
        notes: list[str] = list(resolution_notes)
        if any(seg.get("source_label") == "Generated Main Mix" for seg in main_segments):
            notes.append("This combined transcript came from a generated fallback mix, not a real recorded mix track.")
        dropped = len(main_segments) - len(cleaned)
        if dropped or len(cleaned) != len(merged):
            notes.append(
                f"Dropped {dropped} backchannel micro-segment(s) and merged "
                f"{len(cleaned)} resolved segments into {len(merged)} turns."
            )
        combined_md = format_segments_markdown("Combined Chronological Transcript", merged, notes=notes)
        combined_path.write_text(combined_md, encoding="utf-8")
        written["combined_md"] = str(combined_path)
        log_fn(f"Combined transcript: {combined_path} ({len(merged)} turns)")
    else:
        log_fn("No combined / main-role segments found; skipping combined.md.")

    if extra_segments:
        extras_md = format_segments_markdown("Extra Reference Tracks", extra_segments)
        extras_path.write_text(extras_md, encoding="utf-8")
        written["extras_md"] = str(extras_path)
        log_fn(f"Extras transcript: {extras_path}")

    cleanup_prompt_path.write_text(CLEANUP_PROMPT_TEMPLATE, encoding="utf-8")
    written["cleanup_prompt"] = str(cleanup_prompt_path)

    write_cleanup_bundle(
        bundle_path,
        sections=[
            ("SPEAKER A (solo)", speaker_a_path if speaker_a_segments else None),
            ("SPEAKER B (solo)", speaker_b_path if speaker_b_segments else None),
            ("COMBINED (chronological, Speaker A/B resolved)", combined_path if main_segments else None),
            ("EXTRA REFERENCES", extras_path if extra_segments else None),
        ],
    )
    written["ai_cleanup_bundle"] = str(bundle_path)
    log_fn(f"AI cleanup bundle: {bundle_path}")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Transcription pipeline
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_extracted_tracks(
    session_config_path: Path,
    main_model: str,
    reference_model: str,
    prompt: str,
    log_fn: Callable[[str], None],
    progress_fn: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    ffprobe_path = find_tool("ffprobe")
    client = load_openai_client()

    session_config = json.loads(session_config_path.read_text(encoding="utf-8"))
    session_dir = Path(session_config["session_dir"])
    tracks = session_config.get("tracks", [])
    if not tracks:
        raise RuntimeError("No extracted tracks found in session_config.json.")

    transcripts_dir = session_dir / "transcripts"
    raw_dir = session_dir / "raw_json"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = sum(len(track.get("chunk_files", [])) for track in tracks)
    done_chunks = 0

    all_segments: list[dict[str, Any]] = []
    track_results: list[dict[str, Any]] = []

    for track in tracks:
        label = str(track.get("label") or "Speaker")
        role = str(track.get("role") or ROLE_REFERENCE)
        track_slug = str(track.get("track_slug") or slugify(label))
        chunk_files = [Path(p) for p in track.get("chunk_files", [])]
        if not chunk_files:
            log_fn(f"Skipping {label}: no chunks found.")
            continue

        selected_model = main_model if role in MAIN_ROLES else reference_model
        selected_prompt = "" if selected_model == "gpt-4o-transcribe-diarize" else prompt

        log_fn(f"Transcribing {label} [{role}] with {selected_model}...")
        if selected_model == "gpt-4o-transcribe-diarize" and prompt.strip():
            log_fn("  Note: diarize model does not use prompt; prompt ignored for this track.")

        raw_path = raw_dir / f"{track_slug}_raw.json"
        segments_path = raw_dir / f"{track_slug}_segments.json"

        # Resume-aware: load chunks saved by a previous crashed run and skip ahead.
        raw_responses: list[Any] = []
        track_segments: list[dict[str, Any]] = []
        cumulative_offset = 0.0
        if raw_path.exists() and segments_path.exists():
            try:
                raw_responses = json.loads(raw_path.read_text(encoding="utf-8"))
                track_segments = json.loads(segments_path.read_text(encoding="utf-8"))
                if raw_responses:
                    last = raw_responses[-1]
                    cumulative_offset = float(last.get("offset_seconds", 0.0)) + float(last.get("duration_seconds", 0.0))
                all_segments.extend(track_segments)
                done_chunks += len(raw_responses)
                log_fn(
                    f"  Resuming {label}: {len(raw_responses)}/{len(chunk_files)} chunks already saved, "
                    f"picking up from chunk {len(raw_responses) + 1}."
                )
            except Exception as exc:
                log_fn(f"  Could not resume {label} ({exc}); starting fresh.")
                raw_responses = []
                track_segments = []
                cumulative_offset = 0.0

        already_done = len(raw_responses)

        for idx, chunk_path in enumerate(chunk_files, start=1):
            if idx <= already_done:
                continue
            log_fn(f"  -> {label}: chunk {idx}/{len(chunk_files)}")
            transcript = transcribe_chunk(client, chunk_path, selected_model, selected_prompt)
            plain = to_plain_data(transcript)
            chunk_duration = get_duration_seconds(chunk_path, ffprobe_path) or 0.0

            raw_responses.append(
                {
                    "chunk_index": idx,
                    "chunk_file": str(chunk_path),
                    "offset_seconds": cumulative_offset,
                    "duration_seconds": chunk_duration,
                    "model": selected_model,
                    "role": role,
                    "response": plain,
                }
            )

            chunk_text = extract_text(transcript)
            segments = extract_segments(transcript)
            if segments:
                for seg in segments:
                    text = str(get_attr_or_key(seg, "text", "")).strip()
                    if not text:
                        continue
                    start = float(get_attr_or_key(seg, "start", 0.0) or 0.0) + cumulative_offset
                    end = float(get_attr_or_key(seg, "end", 0.0) or 0.0) + cumulative_offset
                    raw_speaker = str(get_attr_or_key(seg, "speaker", "") or "").strip()
                    speaker = resolve_speaker_tag(role, raw_speaker, label)
                    seg_data = {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": text,
                        "role": role,
                        "source_track": track_slug,
                        "source_label": label,
                        "chunk_index": idx,
                        "model": selected_model,
                    }
                    if raw_speaker and raw_speaker != speaker:
                        seg_data["raw_speaker"] = raw_speaker
                    track_segments.append(seg_data)
                    all_segments.append(seg_data)
            elif chunk_text:
                start = cumulative_offset
                end = cumulative_offset + (chunk_duration or 0.0)
                speaker = resolve_speaker_tag(role, "", label)
                seg_data = {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": chunk_text,
                    "role": role,
                    "source_track": track_slug,
                    "source_label": label,
                    "chunk_index": idx,
                    "model": selected_model,
                    "note": "No segment timestamps returned by selected model; using whole chunk timing.",
                }
                track_segments.append(seg_data)
                all_segments.append(seg_data)

            cumulative_offset += chunk_duration if chunk_duration > 0 else int(session_config.get("chunk_seconds", 540))
            done_chunks += 1
            if progress_fn and total_chunks:
                progress_fn(int(done_chunks / total_chunks * 100))

            # Save after each chunk so a mid-track crash cannot lose completed API calls.
            raw_path.write_text(json.dumps(raw_responses, ensure_ascii=False, indent=2), encoding="utf-8")
            segments_path.write_text(json.dumps(track_segments, ensure_ascii=False, indent=2), encoding="utf-8")

        track_results.append(
            {
                "label": label,
                "role": role,
                "track_slug": track_slug,
                "model": selected_model,
                "raw_json": str(raw_path),
                "segments_json": str(segments_path),
            }
        )

    written = build_session_outputs(session_dir, all_segments, log_fn)

    all_segments.sort(key=lambda item: (float(item.get("start", 0.0)), str(item.get("speaker", ""))))
    (raw_dir / "all_segments.json").write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "session_dir": str(session_dir),
        "transcripts_dir": str(transcripts_dir),
        "track_results": track_results,
        **written,
    }
    summary_path = session_dir / "transcription_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_fn("Transcription complete.")
    return summary


def reprocess_from_raw_json(
    session_dir: Path,
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Regenerate speaker_a.md / speaker_b.md / combined.md / ai_cleanup_bundle.md
    from existing raw_json/*_segments.json without calling OpenAI.
    """
    session_config_path = session_dir / "session_config.json"
    if not session_config_path.exists():
        raise FileNotFoundError(f"session_config.json not found in {session_dir}")

    session_config = json.loads(session_config_path.read_text(encoding="utf-8"))
    tracks = session_config.get("tracks", [])
    if not tracks:
        raise RuntimeError("session_config.json has no tracks.")

    raw_dir = session_dir / "raw_json"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_json folder not found in {session_dir}")

    all_segments: list[dict[str, Any]] = []
    role_by_slug = {str(t.get("track_slug")): str(t.get("role") or ROLE_REFERENCE) for t in tracks}
    label_by_slug = {str(t.get("track_slug")): str(t.get("label") or "Speaker") for t in tracks}

    for track in tracks:
        track_slug = str(track.get("track_slug") or "")
        if not track_slug:
            continue
        seg_file = raw_dir / f"{track_slug}_segments.json"
        if not seg_file.exists():
            log_fn(f"Skipping {track_slug}: {seg_file.name} not found.")
            continue

        try:
            segments = json.loads(seg_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log_fn(f"Failed to read {seg_file.name}: {exc}")
            continue

        role = role_by_slug.get(track_slug, ROLE_REFERENCE)
        label = label_by_slug.get(track_slug, "Speaker")

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            raw_speaker = str(seg.get("speaker", "") or "").strip()
            new_speaker = resolve_speaker_tag(role, raw_speaker, label)
            seg_copy = dict(seg)
            seg_copy["role"] = role
            seg_copy["source_track"] = seg.get("source_track") or track_slug
            seg_copy["source_label"] = seg.get("source_label") or label
            seg_copy["speaker"] = new_speaker
            if raw_speaker and raw_speaker != new_speaker:
                seg_copy["raw_speaker"] = raw_speaker
            all_segments.append(seg_copy)

        log_fn(f"Loaded {len(segments)} segments from {seg_file.name} [{role}].")

    if not all_segments:
        raise RuntimeError("No segments were loaded; nothing to reprocess.")

    written = build_session_outputs(session_dir, all_segments, log_fn)

    all_segments.sort(key=lambda item: (float(item.get("start", 0.0)), str(item.get("speaker", ""))))
    (raw_dir / "all_segments.json").write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "session_dir": str(session_dir),
        "transcripts_dir": str(session_dir / "transcripts"),
        "reprocessed": True,
        **written,
    }
    summary_path = session_dir / "transcription_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_fn("Reprocess complete.")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mode: str, payload: dict[str, Any]) -> None:
        super().__init__()
        self.mode = mode
        self.payload = payload
        self._log_path: Path | None = None

    def write_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.log.emit(line)
        if self._log_path:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def run(self) -> None:
        try:
            ensure_base_dirs()
            if self.mode == "extract":
                result = self.run_extract()
            elif self.mode == "transcribe":
                result = self.run_transcribe()
            elif self.mode == "full":
                extract_result = self.run_extract()
                self.payload["session_config_path"] = extract_result["session_config_path"]
                transcribe_result = self.run_transcribe()
                result = {**extract_result, **transcribe_result}
            elif self.mode == "reprocess":
                result = self.run_reprocess()
            else:
                raise ValueError(f"Unknown worker mode: {self.mode}")

            self.progress.emit(100)
            self.done.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def run_extract(self) -> dict[str, Any]:
        input_path = Path(self.payload["input_path"])
        tracks = self.payload["tracks"]
        chunk_seconds = int(self.payload.get("chunk_seconds", 540))
        clear_old_chunks = bool(self.payload.get("clear_old_chunks", True))
        create_fallback_mix = bool(self.payload.get("create_fallback_mix", False))
        force_fallback_mix = bool(self.payload.get("force_fallback_mix", False))

        session_dir = build_session_dir(input_path)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = session_dir / "processing_log.txt"

        self.write_log(f"Input: {input_path}")
        self.write_log(f"Session folder: {session_dir}")
        self.write_log(f"Chunk length: {chunk_seconds} seconds")

        extracted_tracks: list[dict[str, Any]] = []
        selected_main_tracks = [track for track in tracks if str(track.get("role")) in MAIN_ROLES]
        selected_reference_tracks = [track for track in tracks if str(track.get("role")) in REFERENCE_ROLES]

        total = len(tracks)
        for i, track in enumerate(tracks, start=1):
            extracted = extract_track_to_chunks(
                input_path=input_path,
                track=track,
                session_dir=session_dir,
                chunk_seconds=chunk_seconds,
                clear_old_chunks=clear_old_chunks,
                log_fn=self.write_log,
            )
            extracted_tracks.append(extracted)
            if total:
                self.progress.emit(int(i / max(total, 1) * 70))

        should_create_mix = force_fallback_mix or (create_fallback_mix and not selected_main_tracks)
        if should_create_mix:
            if len(selected_reference_tracks) >= 2:
                generated = create_fallback_mix_chunks(
                    input_path=input_path,
                    source_tracks=selected_reference_tracks,
                    session_dir=session_dir,
                    chunk_seconds=chunk_seconds,
                    clear_old_chunks=clear_old_chunks,
                    log_fn=self.write_log,
                )
                extracted_tracks.insert(0, generated)
            else:
                self.write_log("Fallback mix requested, but fewer than two reference tracks were selected. Skipping mix.")

        session_config = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(input_path),
            "session_dir": str(session_dir),
            "chunk_seconds": chunk_seconds,
            "tracks": extracted_tracks,
        }
        session_config_path = session_dir / "session_config.json"
        session_config_path.write_text(json.dumps(session_config, ensure_ascii=False, indent=2), encoding="utf-8")

        self.write_log("Extraction complete.")
        self.write_log(f"Session config: {session_config_path}")

        return {
            "session_dir": str(session_dir),
            "session_config_path": str(session_config_path),
            "extracted_tracks": extracted_tracks,
        }

    def run_transcribe(self) -> dict[str, Any]:
        session_config_path = Path(self.payload["session_config_path"])
        main_model = str(self.payload.get("main_model") or "gpt-4o-transcribe-diarize")
        reference_model = str(self.payload.get("reference_model") or "gpt-4o-transcribe")
        prompt = str(self.payload.get("prompt") or "")

        session_dir = Path(json.loads(session_config_path.read_text(encoding="utf-8"))["session_dir"])
        self._log_path = session_dir / "processing_log.txt"

        self.write_log(f"Main ordered model: {main_model}")
        self.write_log(f"Reference model: {reference_model}")

        return transcribe_extracted_tracks(
            session_config_path=session_config_path,
            main_model=main_model,
            reference_model=reference_model,
            prompt=prompt,
            log_fn=self.write_log,
            progress_fn=self.progress.emit,
        )

    def run_reprocess(self) -> dict[str, Any]:
        session_dir = Path(self.payload["session_dir"])
        self._log_path = session_dir / "processing_log.txt"
        self.write_log(f"Reprocessing transcripts from raw JSON in: {session_dir}")
        return reprocess_from_raw_json(session_dir=session_dir, log_fn=self.write_log)


# ─────────────────────────────────────────────────────────────────────────────
# PyQt6 GUI
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} V{APP_VERSION}")
        self.resize(1250, 800)
        apply_app_icon(self)

        ensure_base_dirs()

        self.tracks: list[AudioTrack] = []
        self.last_session_config_path: Path | None = None
        self.last_session_dir: Path | None = None
        self.worker: PipelineWorker | None = None

        root = QWidget()
        main_layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        self.full_pipeline_btn = QPushButton("Run Full Pipeline")
        self.full_pipeline_btn.clicked.connect(self.run_full_pipeline)
        self.reprocess_btn = QPushButton("Reprocess From Raw JSON")
        self.reprocess_btn.setToolTip(
            "Regenerate speaker_a.md / speaker_b.md / combined.md / ai_cleanup_bundle.md "
            "from raw_json/*_segments.json without calling OpenAI again."
        )
        self.reprocess_btn.clicked.connect(self.reprocess_existing_session)
        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)
        top_bar.addWidget(self.full_pipeline_btn)
        top_bar.addWidget(self.reprocess_btn)
        top_bar.addWidget(self.open_output_btn)
        top_bar.addStretch(1)
        main_layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_input_tab(), "1. Input / Roles")
        self.tabs.addTab(self.build_extract_tab(), "2. Extract Audio")
        self.tabs.addTab(self.build_transcribe_tab(), "3. Transcribe")
        self.tabs.addTab(self.build_output_tab(), "4. Output / Log")
        main_layout.addWidget(self.tabs)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.setCentralWidget(root)
        self.log_line(f"Data root: {data_root()}")
        self.log_line(f"Tools folder: {tools_dir()}")

    # ── UI construction ─────────────────────────────────────────────────────
    def build_input_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        file_box = QGroupBox("Input file")
        file_layout = QHBoxLayout(file_box)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Choose .mkv/.mp3/.mp4/.wav...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input_file)
        scan_btn = QPushButton("Scan Audio Tracks")
        scan_btn.clicked.connect(self.scan_tracks)
        file_layout.addWidget(self.input_path_edit, 1)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(scan_btn)
        layout.addWidget(file_box)

        self.tracks_table = QTableWidget(0, 9)
        self.tracks_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Track",
                "Map",
                "Codec",
                "Channels",
                "Sample Rate",
                "Title / Language",
                "Role",
                "Speaker Label",
            ]
        )
        header = self.tracks_table.horizontalHeader()
        for col in [0, 1, 2, 3, 4, 5, 7, 8]:
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tracks_table, 1)

        hint = QLabel(
            "Conventions: solo tracks are tagged \"Speaker A\" and \"Speaker B\". "
            "The combined / main track's diarization labels are resolved per chunk by matching content "
            "against the Speaker A and Speaker B solos, so combined.md comes out with proper "
            "\"Speaker A\" / \"Speaker B\" labels directly. Typical 5-track OBS recording: "
            "Track 5 = Combined / Mixed, Track 3 = Speaker B, Track 4 = Speaker A. Reassign as needed."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return tab

    def build_extract_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        options_box = QGroupBox("Extraction options")
        form = QFormLayout(options_box)
        self.chunk_seconds_spin = QSpinBox()
        self.chunk_seconds_spin.setRange(60, 600)
        self.chunk_seconds_spin.setValue(540)
        self.chunk_seconds_spin.setSuffix(" seconds")
        self.clear_old_chunks_check = QCheckBox("Clear old chunks in this session folder before extracting")
        self.clear_old_chunks_check.setChecked(True)
        self.create_fallback_mix_check = QCheckBox(
            "If no Main Ordered track is selected, create a fallback mix from selected reference tracks"
        )
        self.create_fallback_mix_check.setChecked(True)
        form.addRow("Chunk length:", self.chunk_seconds_spin)
        form.addRow("Cleanup:", self.clear_old_chunks_check)
        form.addRow("Fallback mix:", self.create_fallback_mix_check)
        layout.addWidget(options_box)

        button_row = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Selected Tracks")
        self.extract_btn.clicked.connect(self.extract_selected_tracks)
        self.mix_extract_btn = QPushButton("Force Fallback Mix + Extract")
        self.mix_extract_btn.clicked.connect(self.force_fallback_mix_extract)
        button_row.addWidget(self.extract_btn)
        button_row.addWidget(self.mix_extract_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        info = QLabel(
            "Extraction creates 16 kHz mono WAV chunks under DATA/AudioToText/Extracts/[session]/extracted_audio/. "
            "WAV avoids MP3 compression loss. 540 seconds is kept under the typical upload-size limit for mono 16 kHz PCM WAV."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)
        return tab

    def build_transcribe_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_box = QGroupBox("OpenAI transcription")
        form = QFormLayout(model_box)
        self.main_model_combo = QComboBox()
        self.main_model_combo.addItems(
            [
                "gpt-4o-transcribe-diarize",
                "whisper-1",
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
            ]
        )
        self.main_model_combo.setCurrentText("gpt-4o-transcribe-diarize")
        form.addRow("Main ordered track model:", self.main_model_combo)

        self.reference_model_combo = QComboBox()
        self.reference_model_combo.addItems(
            [
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
                "whisper-1",
                "gpt-4o-transcribe-diarize",
            ]
        )
        self.reference_model_combo.setCurrentText("gpt-4o-transcribe")
        form.addRow("Solo reference model:", self.reference_model_combo)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_PROMPT)
        self.prompt_edit.setMinimumHeight(130)
        form.addRow("Prompt/context for prompt-supported models:", self.prompt_edit)
        layout.addWidget(model_box)

        note = QLabel(
            "Recommended setup: use gpt-4o-transcribe-diarize for the combined / mixed track so the "
            "conversation comes out in order with per-turn speaker labels. Use gpt-4o-transcribe for the "
            "solo reference tracks because it accepts the prompt and tends to produce cleaner wording. "
            "Solo tracks are always tagged \"Speaker A\" / \"Speaker B\" regardless of model output; "
            "the combined track's diarization labels are resolved against the solos per chunk."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.transcribe_btn = QPushButton("Send Extracted Chunks to OpenAI")
        self.transcribe_btn.clicked.connect(self.transcribe_current_session)
        layout.addWidget(self.transcribe_btn)
        layout.addStretch(1)
        return tab

    def build_output_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.output_label = QLabel("No output folder yet.")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)
        return tab

    # ── UI actions ──────────────────────────────────────────────────────────
    def browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose audio/video file",
            str(get_app_dir()),
            "Audio/Video (*.mkv *.mp3 *.mp4 *.m4a *.wav *.webm *.mpeg *.mpga *.oga *.ogg);;All files (*.*)",
        )
        if path:
            self.input_path_edit.setText(path)
            self.last_session_config_path = None
            self.last_session_dir = build_session_dir(Path(path))
            self.output_label.setText(f"Output folder will be:\n{self.last_session_dir}")
            self.open_output_btn.setEnabled(True)

    def input_path(self) -> Path:
        raw = self.input_path_edit.text().strip().strip('"').strip("'")
        if not raw:
            raise ValueError("Choose an input file first.")
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found:\n{path}")
        if path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
            raise ValueError(f"Unsupported input extension: {path.suffix}")
        return path

    def scan_tracks(self) -> None:
        try:
            path = self.input_path()
            self.tracks = scan_audio_tracks(path)
            self.populate_tracks_table()
            self.last_session_dir = build_session_dir(path)
            self.output_label.setText(f"Output folder:\n{self.last_session_dir}")
            self.open_output_btn.setEnabled(True)
            self.log_line(f"Detected {len(self.tracks)} audio track(s).")
            if not self.tracks:
                QMessageBox.warning(self, "No audio tracks", "No audio tracks were detected in this file.")
        except Exception as exc:
            QMessageBox.critical(self, "Track scan failed", str(exc))
            self.log_line(f"ERROR: {exc}")

    def default_role_label_for_track(self, track: AudioTrack, total: int) -> tuple[bool, str, str]:
        pos = track.audio_position
        if total >= 5:
            if pos == 4:
                return True, ROLE_MAIN, "Combined Mix"
            if pos == 2:
                return True, ROLE_SPEAKER_B, "Speaker B"
            if pos == 3:
                return True, ROLE_SPEAKER_A, "Speaker A"
            return False, ROLE_IGNORE, f"Track {pos + 1}"
        if total >= 4:
            if pos == 2:
                return True, ROLE_SPEAKER_B, "Speaker B"
            if pos == 3:
                return True, ROLE_SPEAKER_A, "Speaker A"
            return False, ROLE_IGNORE, f"Track {pos + 1}"
        if total == 2:
            if pos == 0:
                return True, ROLE_SPEAKER_A, "Speaker A"
            if pos == 1:
                return True, ROLE_SPEAKER_B, "Speaker B"
        if total == 1:
            return True, ROLE_MAIN, "Combined Audio"
        return False, ROLE_IGNORE, f"Track {pos + 1}"

    def populate_tracks_table(self) -> None:
        self.tracks_table.setRowCount(0)
        total = len(self.tracks)

        for row, track in enumerate(self.tracks):
            self.tracks_table.insertRow(row)
            default_checked, default_role, default_label = self.default_role_label_for_track(track, total)

            use_check = QCheckBox()
            use_check.setObjectName("use_track_check")
            use_check.setChecked(default_checked)
            self.tracks_table.setCellWidget(row, 0, use_check)

            self.tracks_table.setItem(row, 1, QTableWidgetItem(f"Track {track.audio_position + 1}"))
            self.tracks_table.setItem(row, 2, QTableWidgetItem(track.map_spec))
            self.tracks_table.setItem(row, 3, QTableWidgetItem(track.codec))
            self.tracks_table.setItem(row, 4, QTableWidgetItem(track.channels))
            self.tracks_table.setItem(row, 5, QTableWidgetItem(track.sample_rate))
            title_lang = " / ".join(x for x in [track.title, track.language] if x)
            self.tracks_table.setItem(row, 6, QTableWidgetItem(title_lang))

            role_combo = QComboBox()
            role_combo.addItems(ROLE_OPTIONS)
            role_combo.setCurrentText(default_role)
            self.tracks_table.setCellWidget(row, 7, role_combo)

            label_edit = QLineEdit()
            label_edit.setText(default_label)
            self.tracks_table.setCellWidget(row, 8, label_edit)

    def selected_tracks_payload(self) -> list[dict[str, Any]]:
        if not self.tracks:
            raise ValueError("Scan audio tracks first.")

        selected: list[dict[str, Any]] = []
        main_count = 0
        for row, track in enumerate(self.tracks):
            check = self.tracks_table.cellWidget(row, 0)
            role_widget = self.tracks_table.cellWidget(row, 7)
            label_widget = self.tracks_table.cellWidget(row, 8)

            if not (isinstance(check, QCheckBox) and check.isChecked()):
                continue

            role = ROLE_REFERENCE
            if isinstance(role_widget, QComboBox):
                role = role_widget.currentText().strip()
            if role == ROLE_IGNORE:
                continue

            label = ""
            if isinstance(label_widget, QLineEdit):
                label = label_widget.text().strip()

            if role in MAIN_ROLES:
                main_count += 1

            selected.append(
                {
                    "audio_position": track.audio_position,
                    "stream_index": track.stream_index,
                    "map_spec": track.map_spec,
                    "codec": track.codec,
                    "channels": track.channels,
                    "sample_rate": track.sample_rate,
                    "title": track.title,
                    "language": track.language,
                    "role": role,
                    "label": label or f"Track {track.audio_position + 1}",
                }
            )

        if not selected:
            raise ValueError("Select at least one audio track with a non-Ignore role.")
        if main_count > 1:
            raise ValueError("Choose only one Main Ordered / Mixed track. Use reference roles for the solo tracks.")
        return selected

    def base_payload(self, force_fallback_mix: bool = False) -> dict[str, Any]:
        path = self.input_path()
        if not self.tracks:
            self.tracks = scan_audio_tracks(path)
            self.populate_tracks_table()
        return {
            "input_path": str(path),
            "tracks": self.selected_tracks_payload(),
            "chunk_seconds": self.chunk_seconds_spin.value(),
            "clear_old_chunks": self.clear_old_chunks_check.isChecked(),
            "create_fallback_mix": self.create_fallback_mix_check.isChecked(),
            "force_fallback_mix": force_fallback_mix,
            "main_model": self.main_model_combo.currentText(),
            "reference_model": self.reference_model_combo.currentText(),
            "prompt": self.prompt_edit.toPlainText(),
        }

    def extract_selected_tracks(self) -> None:
        try:
            payload = self.base_payload(force_fallback_mix=False)
            self.start_worker("extract", payload)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot extract", str(exc))
            self.log_line(f"ERROR: {exc}")

    def force_fallback_mix_extract(self) -> None:
        try:
            payload = self.base_payload(force_fallback_mix=True)
            self.start_worker("extract", payload)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot create fallback mix", str(exc))
            self.log_line(f"ERROR: {exc}")

    def transcribe_current_session(self) -> None:
        try:
            if not self.last_session_config_path:
                path = self.input_path()
                maybe_config = build_session_dir(path) / "session_config.json"
                if maybe_config.exists():
                    self.last_session_config_path = maybe_config
                else:
                    raise ValueError("Extract audio first, or run the full pipeline.")

            payload = {
                "session_config_path": str(self.last_session_config_path),
                "main_model": self.main_model_combo.currentText(),
                "reference_model": self.reference_model_combo.currentText(),
                "prompt": self.prompt_edit.toPlainText(),
            }
            self.start_worker("transcribe", payload)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot transcribe", str(exc))
            self.log_line(f"ERROR: {exc}")

    def run_full_pipeline(self) -> None:
        try:
            payload = self.base_payload(force_fallback_mix=False)
            self.start_worker("full", payload)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot run pipeline", str(exc))
            self.log_line(f"ERROR: {exc}")

    def reprocess_existing_session(self) -> None:
        try:
            start_dir = self.last_session_dir or extracts_root()
            chosen = QFileDialog.getExistingDirectory(
                self,
                "Choose a session folder (must contain session_config.json and raw_json/)",
                str(start_dir),
            )
            if not chosen:
                return
            session_dir = Path(chosen)
            if not (session_dir / "session_config.json").exists():
                raise FileNotFoundError(
                    f"session_config.json not found in:\n{session_dir}\n\n"
                    "Pick the session folder created by a previous run."
                )
            self.last_session_dir = session_dir
            self.last_session_config_path = session_dir / "session_config.json"
            self.output_label.setText(f"Output folder:\n{session_dir}")
            self.open_output_btn.setEnabled(True)
            self.start_worker("reprocess", {"session_dir": str(session_dir)})
        except Exception as exc:
            QMessageBox.critical(self, "Cannot reprocess", str(exc))
            self.log_line(f"ERROR: {exc}")

    def start_worker(self, mode: str, payload: dict[str, Any]) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        self.progress.setValue(0)
        self.set_busy(True)
        self.tabs.setCurrentIndex(3)
        self.log_line(f"Starting: {mode}")

        self.worker = PipelineWorker(mode, payload)
        self.worker.log.connect(self.log_line)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self.worker_done)
        self.worker.error.connect(self.worker_error)
        self.worker.start()

    def worker_done(self, result: dict[str, Any]) -> None:
        self.set_busy(False)
        if result.get("session_config_path"):
            self.last_session_config_path = Path(result["session_config_path"])
        if result.get("session_dir"):
            self.last_session_dir = Path(result["session_dir"])
            self.output_label.setText(f"Output folder:\n{self.last_session_dir}")
            self.open_output_btn.setEnabled(True)
        self.log_line("Done.")
        QMessageBox.information(self, "Done", "AudioToText job finished.")

    def worker_error(self, message: str) -> None:
        self.set_busy(False)
        self.log_line(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)

    def set_busy(self, busy: bool) -> None:
        self.full_pipeline_btn.setEnabled(not busy)
        self.reprocess_btn.setEnabled(not busy)
        self.extract_btn.setEnabled(not busy)
        self.mix_extract_btn.setEnabled(not busy)
        self.transcribe_btn.setEnabled(not busy)

    def open_output_folder(self) -> None:
        try:
            if self.last_session_dir:
                open_folder(self.last_session_dir)
            else:
                open_folder(extracts_root())
        except Exception as exc:
            QMessageBox.critical(self, "Could not open folder", str(exc))

    def log_line(self, line: str) -> None:
        self.log_box.appendPlainText(line)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_app_icon(app)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
