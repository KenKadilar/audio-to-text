# AudioToText

Splits a multi-track audio recording (e.g. an OBS `.mkv` where each speaker has their own track) into clean, speaker-labeled transcripts. The interesting bit is a per-chunk speaker-resolution step that fixes a sharp diarization failure mode.

<img width="1243" height="821" alt="Audio-To-Text" src="https://github.com/user-attachments/assets/86ffa044-9873-4d79-814f-dae6706a2dbb" />

## Features

- **Multi-track audio split**: ffmpeg pulls each track out into ~9-minute WAV chunks (auto-synthesizes a combined mix if the recording doesn't have one).
- **Speaker-labeled timeline**: diarizer output on the combined track is stitched to "Speaker A" / "Speaker B" by cross-referencing the solo tracks per chunk.
- **Works with one solo track**: when only one side was captured separately (a call app's audio on its own track), the best per-chunk match is tagged as that speaker and the rest of the chunk is assigned by elimination.
- **Cache-based reprocess**: re-render the output transcripts from cached responses without re-spending API credits.
- **Resume-aware**: each chunk is saved as it lands, so an interrupted run picks up where it stopped instead of re-paying for finished work.
- **Review outputs**: a compact `ai_review.md` (chunk verdicts, unresolved labels, timeline) plus `ai_review.json` with the same content as structured turns.

## Tech stack

Python 3.11+, PyQt6, OpenAI Audio Transcription API, ffmpeg/ffprobe.

## Install + run

```powershell
pip install -r requirements.txt
python AudioToText.py
```

Needs `ffmpeg`/`ffprobe` on `PATH` (or dropped into `DATA/TOOLS/`) and `DATA/AudioToText/.env` with `OPENAI_API_KEY=...`.

GUI flow: scan a recording's audio tracks -> assign roles (Combined / Mixed, Speaker A, Speaker B, Ignore) -> extract -> transcribe. "Run Full Pipeline" does both steps end-to-end; "Reprocess From Raw JSON" rebuilds outputs from cached responses.

Outputs land under `DATA/AudioToText/Extracts/<session>/transcripts/`: per-speaker solos, a chronological combined timeline, and the two review files.
