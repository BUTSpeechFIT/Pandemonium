"""Gradio preview of Pandemonium multi-talker ASR (access-code gated)."""

import asyncio
import functools
import re
import html
import json
import os
import subprocess
import tempfile
import threading
import time

# Gradio's default /tmp/gradio is shared by every user of the machine and fails on a multi-user
# box once someone else has created it; keep ours per user unless told otherwise.
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), f"gradio-{os.getuid()}"))

import gradio as gr
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from librosa import load as librosa_load

import access
from example import load_example
from pandemonium import SAMPLE_RATE
from pipeline import DemoPipeline, Diarization, utterances
from remote import GPUPool, configured_remotes
from timeline import CSS as TIMELINE_CSS, JS as TIMELINE_JS, PLAY_ON_CLICK_JS, audio_url, render_timeline

# Advertised as 30 minutes, but up to 40 are accepted, so recordings that run a little over still
# go through. Only a refused file is told the real number. Bounded by the shared 16 GB GPU, not the model.
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "2400"))
LIMIT = access.LIMIT  # every user-facing mention of the limit
MIN_SECONDS = 1
# Audio only. Listed explicitly as well, because browsers label WebM voice recordings video/webm,
# which an "audio" filter alone would refuse; a file that really carries video is rejected on upload.
ACCEPTED_EXTENSIONS = [".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".webm", ".weba",
                       ".wma", ".amr", ".aif", ".aiff"]
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#db2777", "#65a30d"]

pipeline = DemoPipeline()
# Jobs run here when this GPU is free, else on a remote one (remote.py), else they wait.
gpus = GPUPool(pipeline, configured_remotes())

CSS = """
.pk-transcript { display: flex; flex-direction: column; gap: 1px; max-height: 560px; overflow-y: auto;
                 font-size: 15px; line-height: 1.55; padding-right: 4px; }
.pk-turn { display: grid; grid-template-columns: 4.8em 7.2em 1fr; column-gap: 14px; padding: 5px 10px;
           border-radius: 6px; }
.pk-turn.pk-nospk { grid-template-columns: 4.8em 1fr; }
.pk-turn { cursor: pointer; }
.pk-turn:hover { background: var(--background-fill-secondary); }
/* What the player is on right now (timeline.py highlights it). */
.pk-turn.pk-live { background: color-mix(in srgb, var(--color-accent, #f97316) 13%, transparent);
                   box-shadow: inset 3px 0 0 var(--color-accent, #f97316); }
.pk-cont { padding-top: 1px; }
.pk-cont .pk-spk { visibility: hidden; }
.pk-ts { font-family: var(--font-mono); font-size: 13px; color: var(--body-text-color-subdued);
         font-variant-numeric: tabular-nums; padding-top: 2px; white-space: nowrap; }
.pk-spk { font-weight: 600; color: var(--c); white-space: nowrap; }
.pk-spk::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                  background: var(--c); margin-right: 6px; vertical-align: 1px; }
.pk-section { padding: 6px 0 12px; border-bottom: 1px solid var(--border-color-primary); }
.pk-section:last-child { border-bottom: none; }
.pk-head { display: flex; align-items: baseline; gap: 12px; padding: 4px 10px 6px; cursor: pointer; list-style: none;
           border-radius: 6px; user-select: none; }
.pk-head::-webkit-details-marker { display: none; }
.pk-head:hover { background: var(--background-fill-secondary); }
.pk-caret { display: inline-block; width: 12px; margin-right: -6px; font-size: 11px; color: var(--body-text-color-subdued);
            transition: transform .15s; }
details.pk-section:not([open]) .pk-caret { transform: rotate(-90deg); }
details.pk-section:not([open]) { padding-bottom: 6px; }
.pk-meta { font-size: 13px; color: var(--body-text-color-subdued); }
.pk-notice { padding: 10px 14px; border-radius: 8px; font-size: 15px;
             border: 1px solid color-mix(in srgb, #f59e0b 55%, transparent);
             background: color-mix(in srgb, #f59e0b 12%, var(--background-fill-primary)); }
.pk-notice b { font-weight: 600; }
.pk-account { font-size: 13px; color: var(--body-text-color-subdued); padding: 2px 2px 0; }
.pk-account b { color: var(--body-text-color); }
/* Bold (700) is not among the theme font weights loaded and falls back to another face; 600 is. */
.prose strong, .prose b { font-family: inherit; font-weight: 600; }
.pk-empty { color: var(--body-text-color-subdued); padding: 8px 10px; }
/* Audio is never offered for download: the upload box lists the file without its download link. */
#pk-upload [data-testid=download-link] { display: none; }
.pk-refused { padding: 10px 14px; border-radius: 8px; font-size: 15px; border: 1px solid rgba(220, 38, 38, .45);
              background: color-mix(in srgb, #dc2626 8%, var(--background-fill-primary)); }
/* Gradio pads every link in an HTML block (2px 8px), which leaves gaps around links inside text. */
.pk-empty a, .pk-meta a, .pk-account a, .pk-about a, .pk-expired-card p a { padding: 0; }
.pk-cta { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; padding: 14px 16px; border-radius: 10px;
          font-size: 15px; border: 1px solid color-mix(in srgb, #0e9f6e 45%, transparent);
          background: color-mix(in srgb, #0e9f6e 9%, var(--background-fill-primary)); }
.pk-cta > div:first-child { flex: 1 1 380px; }
.pk-cta b { font-weight: 600; }
.pk-cta-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.pk-btn { display: inline-block; padding: 9px 18px; border-radius: 8px; background: #0e9f6e; color: #fff !important;
          font-weight: 600; text-decoration: none !important; white-space: nowrap; }
.pk-btn-2 { background: transparent; color: #0e9f6e !important; border: 1px solid #0e9f6e; }
.pk-about { display: flex; flex-direction: column; gap: 4px; padding: 2px 2px 0; font-size: 14px; }
.pk-about b { font-weight: 600; }
.pk-downloads { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.pk-downloads .pk-btn { padding: 7px 14px; font-size: 14px; }
.pk-dl-label { font-weight: 600; margin-right: 4px; }
button.pk-copy { border: 0; cursor: pointer; font: inherit; font-weight: 600; }
.pk-ready { padding: 8px 14px; border-radius: 8px; font-size: 14px; border: 1px solid rgba(14, 159, 110, .4);
            background: color-mix(in srgb, #0e9f6e 7%, var(--background-fill-primary)); }
.pk-ready b { font-weight: 600; }
.pk-again { align-items: center; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border-color-primary);
            background: var(--block-background-fill); }
.pk-again-note { font-size: 14px; }
.pk-searchbar { display: flex; align-items: center; gap: 10px; margin: 2px 0 8px; }
.pk-search { flex: 1 1 auto; max-width: 420px; padding: 7px 10px; border-radius: 8px; font: inherit; font-size: 14px;
             border: 1px solid var(--border-color-primary); background: var(--background-fill-primary);
             color: var(--body-text-color); }
.pk-search-count { font-size: 13px; color: var(--body-text-color-subdued); }
.pk-turn.pk-hide { display: none; }
.pk-transcript.filtering .pk-cont .pk-spk { visibility: visible; }
.pk-share { padding: 4px 10px 12px; border-bottom: 1px solid var(--border-color-primary); }
.pk-share-title { font-weight: 600; margin-bottom: 6px; }
.pk-share-row { display: grid; grid-template-columns: 9em 1fr 7.5em; align-items: center; gap: 10px; padding: 2px 0; }
.pk-share-row .pk-spk { overflow: hidden; text-overflow: ellipsis; }
.pk-share-bar { height: 10px; border-radius: 5px; background: var(--background-fill-secondary); overflow: hidden; }
.pk-share-bar i { display: block; height: 100%; border-radius: 5px; opacity: .8; }
.pk-share-row .pk-meta { text-align: right; font-variant-numeric: tabular-nums; }
.pk-rename { margin-left: -4px; padding: 1px 8px; border: 1px solid var(--border-color-primary); border-radius: 6px;
             background: var(--background-fill-primary); cursor: pointer; font-size: 12px;
             color: var(--body-text-color-subdued); }
.pk-rename:hover { color: var(--body-text-color); border-color: var(--body-text-color-subdued); }
.pk-share-tip { margin-top: 8px; }
.pk-merge { margin-left: 2px; padding: 1px 4px; border: 1px solid var(--border-color-primary); border-radius: 6px;
            background: var(--background-fill-primary); color: var(--body-text-color-subdued); font: inherit;
            font-size: 12px; cursor: pointer; }
.pk-unmerge { margin-top: 8px; padding: 4px 12px; border: 1px solid var(--border-color-primary); border-radius: 6px;
              background: var(--background-fill-primary); color: var(--body-text-color); font: inherit; font-size: 13px;
              cursor: pointer; }
.pk-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--body-text-color-subdued);
             padding: 0 2px; }
.pk-legend span { display: inline-flex; align-items: center; gap: 6px; }
.pk-legend i { display: inline-block; width: 22px; height: 12px; border-radius: 3px; }
.pk-lg-block { background: color-mix(in srgb, #2563eb 18%, var(--background-fill-primary)); border-left: 3px solid #2563eb; }
.pk-lg-true { height: 3px !important; background: #2563eb; }
.pk-lg-prob { background: color-mix(in srgb, #2563eb 45%, transparent); height: 8px !important; }
.pk-lg-ref { background: var(--background-fill-primary); border: 1.5px dashed #2563eb; border-left: 3px solid #2563eb; }
@media (max-width: 640px) { .pk-share-row { grid-template-columns: 6.5em 1fr 3em; } .pk-share-time { display: none; } }
.pk-expired { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center;
              padding: 16px; background: rgba(0, 0, 0, .45); }
.pk-expired-card { max-width: 440px; padding: 24px 26px; border-radius: 12px; text-align: center; font-size: 15px;
                   background: var(--background-fill-primary); color: var(--body-text-color);
                   border: 1px solid var(--border-color-primary); box-shadow: 0 10px 30px rgba(0, 0, 0, .25); }
.pk-expired-card b { font-size: 18px; font-weight: 600; }
.pk-expired-card p { margin: 10px 0 16px; }
.pk-expired-card .pk-meta { margin: 14px 0 0; font-size: 13px; color: var(--body-text-color-subdued); }
.pk-bench { box-sizing: border-box; padding: 16px 20px; border-radius: 12px;
             border: 1px solid color-mix(in srgb, #0e9f6e 45%, transparent);
             background: var(--block-background-fill); }
.pk-bench-title { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
.pk-bench table { width: 100%; max-width: 820px; margin: 4px 0 10px; border-collapse: collapse; border: none;
                  font-size: 16px; font-variant-numeric: tabular-nums; }
.pk-bench th, .pk-bench td { padding: 7px 12px; border: none; border-bottom: 1px solid var(--border-color-primary);
                             text-align: right; white-space: nowrap; background: none; vertical-align: middle; }
.pk-bench tr { border: none; }  /* Gradio's prose styles box every row */
.pk-bench th:first-child, .pk-bench td:first-child { text-align: left; padding-left: 0; white-space: normal; }
.pk-bench th { font-weight: 600; color: var(--body-text-color-subdued); font-size: 13px; line-height: 1.3;
               vertical-align: bottom; }
.pk-bench th small { font-weight: 500; }
.pk-bench th a { padding: 0; color: inherit; }
.pk-bench td.pk-best { font-weight: 700; color: var(--body-text-color); }
.pk-bench tr.pk-sum td { border-top: 1px solid var(--body-text-color-subdued); font-weight: 600; }
.pk-bench tr.pk-group td { padding: 14px 0 6px; text-align: left; white-space: normal; }
.pk-bench tr.pk-group small { margin-left: 8px; font-size: 13px; color: var(--body-text-color-subdued); }
.pk-bench .pk-tag { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
                    letter-spacing: .02em; text-transform: uppercase; }
.pk-bench tr.pk-in .pk-tag { color: #0e9f6e; background: color-mix(in srgb, #0e9f6e 14%, transparent); }
.pk-bench tr.pk-ood .pk-tag { color: #c2410c; background: color-mix(in srgb, #ea580c 15%, transparent); }
.dark .pk-bench tr.pk-in .pk-tag { color: #34d399; }
.dark .pk-bench tr.pk-ood .pk-tag { color: #fb923c; }
.pk-bench tr.pk-ood-row td { background: color-mix(in srgb, #ea580c 5%, transparent); }
.pk-bench tr.pk-ood-row td:first-child { box-shadow: inset 3px 0 0 color-mix(in srgb, #ea580c 55%, transparent);
                                         padding-left: 10px; }
.pk-bench .pk-bench-sub { margin-top: 26px; padding-top: 18px; border-top: 1px solid var(--border-color-primary);
                         max-width: 820px; }
.pk-bench .pk-scroll { max-width: 820px; overflow-x: auto; }
.pk-bench .pk-meta b { font-weight: 600; }
.pk-bench .pk-meta + .pk-meta { margin-top: 4px; }
.pk-bench td a { padding: 0; margin: 0; white-space: nowrap; color: var(--body-text-color); font-weight: 600;
                 text-decoration: underline;
                 text-decoration-color: color-mix(in srgb, #0e9f6e 60%, transparent); text-underline-offset: 3px; }
.pk-bench td a:hover { color: #0e9f6e; }
.pk-bench .pk-meta { font-size: 13px; line-height: 1.5; max-width: 820px; }
.pk-bench tfoot td { font-size: 13px; color: var(--body-text-color-subdued); white-space: normal; text-align: left;
                     border-bottom: none; padding-top: 8px; }
@media (max-width: 640px) { .pk-bench { padding: 12px 12px; overflow-x: auto; }
                            .pk-bench th, .pk-bench td { padding: 6px 4px; }
                            .pk-bench th { font-size: 11.5px; } .pk-bench td { font-size: 14px; }
                            .pk-bench td a { white-space: normal; }
                            .pk-bench .pk-single th { white-space: normal; }
                            .pk-bench table { font-size: 15px; } }
@media (max-width: 640px) {
  .pk-turn { grid-template-columns: 4.2em 1fr; }
  .pk-turn .pk-spk { grid-column: 2; }
  .pk-turn .pk-txt { grid-column: 2; }
  .pk-cont .pk-spk { display: none; }
}
"""


def _name(spk: int, names=None) -> str:
    """A speaker's display name: what the user renamed them to (`names`, keyed by row), else "Speaker N"."""
    return (names or {}).get(str(spk)) or f"Speaker {spk + 1}"


def _parse_merges(text, num_speakers) -> dict:
    """The merges the page sends (#pk-merges: JSON {from: into}), as {row: the row it ends up in}."""
    try:
        raw = json.loads(text or "{}")
    except ValueError:
        return {}
    into = {}
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        try:
            k, v = int(k), int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= k < num_speakers and 0 <= v < num_speakers and k != v:
            into[k] = v

    def root(k, seen=()):  # follow chains (3 -> 1 -> 0); a cycle leaves its first speaker as is
        return k if k not in into or k in seen else root(into[k], seen + (k,))

    return {k: root(k) for k in into if root(k) != k}


def _merged(words, diar, merges):
    """The result with speakers merged: (words per row, diarization, rows no longer shown).

    A merged speaker's words join the other speaker's, the two probability curves combine (the higher at
    each moment), and its own row stays empty and hidden. Rows keep their numbers, so colours and names
    stay put.
    """
    if not merges:
        return words, diar, set()
    out = [list(ws) for ws in words]
    probs = diar.probs.copy()
    for k, r in merges.items():
        out[r] += out[k]
        out[k] = []
        probs[r] = np.maximum(probs[r], diar.probs[k])
        probs[k] = 0
    for ws in out:
        ws.sort(key=lambda w: w.start)
    view = Diarization(probs=probs, frame_shift=diar.frame_shift,
                       segments=sorted((s, e, merges.get(k, k)) for s, e, k in diar.segments))
    view.source = getattr(diar, "source", None)
    return out, view, set(merges)


def _parse_names(text) -> dict:
    """The renames the page sends (#pk-names: JSON {row: name}), cleaned: one line, at most 40 characters."""
    try:
        raw = json.loads(text or "{}")
    except ValueError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(int(k)): " ".join(str(v).split())[:40] for k, v in raw.items()
            if str(k).isdigit() and int(k) < 16 and str(v).strip()}


def _fmt(t: float) -> str:
    """Seconds -> `m:ss.s` (or `h:mm:ss.s`)."""
    m, s = divmod(round(t, 1), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # static build: m4a/AAC, mp3, opus/webm, ...
UNREADABLE = "Could not read this file as audio. Try WAV, FLAC, MP3, M4A, OGG or WebM."
NO_VIDEO = "Video files are not supported: please upload an audio file (WAV, FLAC, MP3, M4A, OGG or WebM)."
# What every major browser can play back natively; anything else gets an MP3 copy for the player.
BROWSER_PLAYABLE = {".wav", ".mp3", ".m4a", ".aac", ".flac"}


@functools.lru_cache(maxsize=8)
def _probe_seconds(path):
    """Duration from the file header, without decoding (soundfile, else ffmpeg's banner).

    Also where video is refused: a real video stream fails, while the cover art MP3/M4A files
    often carry (an "attached pic" stream) does not.
    """
    try:
        return sf.info(path).duration
    except Exception:
        banner = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-i", path], capture_output=True, text=True).stderr
        if any("Video:" in line and "attached pic" not in line for line in banner.splitlines()):
            raise gr.Error(NO_VIDEO)
        m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", banner)
        if not m:
            raise gr.Error(UNREADABLE)
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


@functools.lru_cache(maxsize=2)
def _decode(path):
    """Decoded once per upload: diarize, transcribe and the timeline all reuse it.

    soundfile reads WAV/FLAC/OGG directly; anything else goes through the bundled ffmpeg, since
    librosa no longer falls back to one. Decoding stops just past the length limit either way.
    """
    try:
        audio, _ = librosa_load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS + 1)
    except Exception:
        audio = _ffmpeg_decode(path)
    return audio.astype(np.float32)


@functools.lru_cache(maxsize=4)
def _playback_path(path):
    """A file the browser can play for the timeline: the upload itself, or an MP3 copy of it.

    The copy sits next to the upload, inside Gradio's upload folder, so it is served (and
    gated) exactly like the upload.
    """
    if os.path.splitext(path)[1].lower() in BROWSER_PLAYABLE:
        return path
    out = os.path.splitext(path)[0] + ".playback.mp3"
    if not os.path.exists(out):
        cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-i", path, "-vn", "-t", str(MAX_SECONDS + 1),
               "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k", out]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return path  # fall back to the original; recognition does not depend on this
    return out


def _ffmpeg_decode(path):
    """Any audio file -> mono float32 at SAMPLE_RATE, via ffmpeg."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", path, "-vn", "-t", str(MAX_SECONDS + 1),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise gr.Error(UNREADABLE)
    return np.frombuffer(out.stdout, dtype=np.float32)


def _too_long(seconds):
    return gr.Error(f"This recording is {_fmt(seconds)} long; the preview accepts about {LIMIT} "
                    f"({MAX_SECONDS // 60} minutes at most). Please upload a shorter excerpt.")


def _load(path):
    if not path:  # only API callers get here: the buttons check first (`needs_audio`)
        raise gr.Error(NO_AUDIO)
    # Refuse an over-long file from its header, before spending time decoding it.
    if _probe_seconds(path) > MAX_SECONDS + 0.5:
        raise _too_long(_probe_seconds(path))
    audio = _decode(path)
    seconds = len(audio) / SAMPLE_RATE
    if seconds > MAX_SECONDS + 0.5:
        raise _too_long(seconds)
    if seconds < MIN_SECONDS:
        raise gr.Error("Audio is shorter than a second.")
    return audio


def timeline(path, audio, diar, utts=None, open_at=0.0, reference=None, src=None, names=None, renamable=False,
             hidden=()) -> str:
    labels = [_name(s, names) for s in range(diar.num_speakers)]
    colors = [_color(s) for s in range(diar.num_speakers)]
    return render_timeline(audio, SAMPLE_RATE, diar, labels, colors, utts,
                           src=src or audio_url(_playback_path(path)), open_at=open_at, reference=reference,
                           renamable=renamable, hidden=hidden)


# Who may download which file. Gradio serves any file it created (uploads, results) to any logged-in user
# who knows its path, so the server also checks that it is theirs: every file the app hands someone is
# registered here, and `/gradio_api/file=` answers only its owner (see `_file_allowed`). In memory, like
# the login sessions themselves: a restart ends both.
_owners: dict = {}
# Where the exports are written (inside Gradio's temp folder, so the weekly clean-up covers them too).
RESULTS_DIR = os.path.join(os.environ["GRADIO_TEMP_DIR"], "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _account(username) -> str:
    return access.ADMIN_USER if access.is_admin(username) else access.normalize_email(username or "")


def _own(path, username):
    if path and username:
        _owners[os.path.realpath(path)] = _account(username)


def _purge_account(account: str):
    """After an account is deleted: its files, its remembered results and its open sessions go too."""
    for path, owner in list(_owners.items()):
        if owner == account:
            try:
                os.remove(path)
            except OSError:
                pass
            _owners.pop(path, None)
    for key in [k for k in _results if k[0] == account]:
        _results.pop(key, None)
    _active_cache.pop(account, None)
    if _sessions is not None:
        for token in [t for t, u in _sessions.items() if _account(u) == account]:
            _sessions.pop(token, None)


_sessions = None  # the logged-in app's session tokens, set once it is mounted


def _file_allowed(path: str, username) -> bool:
    return bool(username) and _owners.get(os.path.realpath(path)) == _account(username)


def run_diarization(path, names_json, request: gr.Request, progress=gr.Progress()):
    audio = _load(path)
    refused = access.reserve_diarization(request.username)
    if refused:  # over the hourly cap (or revoked): a warning, the page stays as it is
        gr.Warning(refused)
        return gr.skip(), gr.skip(), gr.skip()
    progress(0.1, desc="Finding speakers…")
    try:
        with gpus.lease() as gpu:
            diar = gpu.diarize(audio)
            _log_job("diarize", gpu, audio)
    except torch.OutOfMemoryError:
        raise gr.Error(GPU_FULL)
    diar.source = path  # which upload this belongs to; see run_transcription
    if diar.num_speakers == 0:  # not an error: say so and leave the page as it is
        gr.Warning(NO_SPEECH)
        return gr.skip(), gr.skip(), gr.skip()
    info = (f"Found **{diar.num_speakers}** speaker(s) in {_fmt(len(audio) / SAMPLE_RATE)}. "
            "The shaded curve in each lane is how likely that speaker is talking; "
            "**Transcribe** fills the lanes with what each one said.")
    _own(_playback_path(path), request.username)  # the player's copy, for formats browsers do not play
    return diar, timeline(path, audio, diar, names=_parse_names(names_json), renamable=True), info


# Each account's recent results, by recording: transcribing the same file again shows the earlier
# result for free, with a "Transcribe afresh" button for when a fresh run is really wanted. Identical
# uploads land at the same path (Gradio names the folder by the file's hash), so a re-upload counts too.
_results: dict = {}
RESULTS_PER_ACCOUNT = 5


def _remember(username, path, entry):
    account = _account(username)
    _results[(account, os.path.realpath(path))] = entry
    mine = sorted((k for k in _results if k[0] == account), key=lambda k: _results[k]["time"])
    for k in mine[:-RESULTS_PER_ACCOUNT]:
        _results.pop(k, None)


def _earlier(username, path):
    return _results.get((_account(username), os.path.realpath(path))) if path else None


def run_transcription(path, diar, names_json, request: gr.Request, progress=gr.Progress()):
    return _run_transcription(path, diar, names_json, request, progress, use_earlier=True)


def run_transcription_again(path, diar, names_json, request: gr.Request, progress=gr.Progress()):
    return _run_transcription(path, diar, names_json, request, progress, use_earlier=False)


def _run_transcription(path, diar, names_json, request, progress, use_earlier):
    audio = _load(path)
    names = _parse_names(names_json)
    earlier = _earlier(request.username, path) if use_earlier else None
    if earlier:  # already done for this person: show it, no transcription used
        return _present(path, audio, earlier, names, request, again=earlier["time"])
    try:
        request_id = access.reserve(request.username, _ip(request), len(audio) / SAMPLE_RATE)
    except access.QuotaError as err:
        raise gr.Error(str(err))
    try:
        entry = _transcribe(path, audio, diar, progress)
    except NoSpeech:
        # Nothing to transcribe is not the user's mistake to pay for, nor an error: a warning, and the
        # page stays as it was (raising, even quietly, would mark the timeline "Error").
        access.refund(request_id)
        gr.Warning(NO_SPEECH)
        return (gr.skip(),) * 9 + (account_status(request), gr.skip(), gr.skip(), gr.skip())
    except Exception as err:
        access.refund(request_id)  # a failed run does not cost the user a transcription
        if isinstance(err, torch.OutOfMemoryError):
            raise gr.Error(GPU_FULL)
        raise
    _remember(request.username, path, entry)
    return _present(path, audio, entry, names, request)


def _ip(request: gr.Request) -> str:
    return access.client_ip(request.headers, request.client.host if request.client else "")


def account_status(request: gr.Request) -> str:
    """"Signed in as ... · N of M transcriptions left · Log out"."""
    user = request.username or ""
    left = access.remaining(user)
    usage = "no limit (admin)" if left is None else f"<b>{left[0]} of {left[1]}</b> transcriptions left"
    return (f"<div class='pk-account'>Signed in as <b>{html.escape(user)}</b> · {usage} · "
            f"<a href='logout'>Log out</a></div>")


def _log_job(what, gpu, audio):
    print(f"{what}: {len(audio) / SAMPLE_RATE:.0f} s on {gpu.where}", flush=True)


def _transcribe(path, audio, diar, progress):
    # Reuse the speakers found for THIS upload only. The state is cleared when the audio changes,
    # but a click racing the upload, or an API client, can still arrive with another file's
    # speakers -- and would transcribe this audio with them.
    with gpus.lease() as gpu:  # both steps on one machine, which then holds the audio already
        if diar is None or getattr(diar, "source", None) != path:
            progress(0.05, desc="Finding speakers…")
            diar = gpu.diarize(audio)
            diar.source = path
        if diar.num_speakers == 0:
            raise NoSpeech  # run_transcription refunds it and warns
        progress(0.3, desc=f"Transcribing {diar.num_speakers} speaker{'s' if diar.num_speakers > 1 else ''}…")
        words = gpu.transcribe(audio, diar)
        _log_job("transcribe", gpu, audio)
    progress(0.95, desc="Preparing the results…")
    return {"time": time.time(), "diar": diar, "words": words, "utts": utterances(words)}


def _ago(t: float) -> str:
    """How long ago, in words: the server's clock time would be in the wrong time zone for many visitors."""
    minutes = int((time.time() - t) // 60)
    if minutes < 1:
        return "a moment ago"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days > 1 else ''} ago"


def _present(path, audio, entry, names, request, again=None):
    """Everything the page shows for a result. `again`: when it was made, if it is an earlier one."""
    diar, words, utts = entry["diar"], entry["words"], entry["utts"]
    conversation = render_conversation(utts, names)
    per_speaker = render_per_speaker(utts, len(words), names, renamable=True, mergeable=True)
    _own(_playback_path(path), request.username)
    plot = timeline(path, audio, diar, utts, names=names, renamable=True)
    note = (f"<div class='pk-again-note'>This is the transcript you made of this recording {_ago(again)}, shown "
            "again: nothing was deducted from your transcriptions. To run the model on it afresh, use the button."
            "</div>") if again else ""
    return (diar, plot, conversation, per_speaker, gr.update(value=_table(utts, names), visible=True), SEGMENTS_HINT,
            utts, words, _downloads(utts, words, names, request), account_status(request),
            gr.update(visible=bool(again)), note, "")  # "": a new result starts unmerged


def _table(utts, names=None):
    return [[_fmt(u["start"]), _fmt(u["end"]), _name(u["speaker"], names), u["text"]] for u in utts]


def _downloads(utts, words, names, request):
    """Write the exports into the account's own folder and offer them (only its owner can fetch them)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_dir = tempfile.mkdtemp(dir=RESULTS_DIR)
    files = _export_files(utts, words, names, out_dir)
    for path in files.values():
        _own(path, request.username)
    urls = [(label, audio_url(files[name])) for name, label in EXPORTS]
    return gr.update(value=_downloads_html(urls, _transcript_text(utts, names)), visible=True)


def apply_names(names_json, merges_json, words, diar, request: gr.Request):
    """The user renamed a speaker: the page did the views itself; redo the table and the downloads."""
    if not words or diar is None:
        return gr.skip(), gr.skip()
    names = _parse_names(names_json)
    merged, _, _ = _merged(words, diar, _parse_merges(merges_json, len(words)))
    utts = utterances(merged)
    return gr.update(value=_table(utts, names)), _downloads(utts, merged, names, request)


def apply_merges(merges_json, names_json, path, words, diar, request: gr.Request):
    """The user merged speakers (or undid it): redraw the result from the original with these merges."""
    if not words or diar is None or not path:
        return (gr.skip(),) * 6
    names = _parse_names(names_json)
    merged, view, hidden = _merged(words, diar, _parse_merges(merges_json, len(words)))
    utts = utterances(merged)
    audio = _load(path)
    return (timeline(path, audio, view, utts, names=names, renamable=True, hidden=hidden),
            render_conversation(utts, names),
            render_per_speaker(utts, len(words), names, renamable=True, mergeable=True, hidden=hidden),
            gr.update(value=_table(utts, names)), utts, _downloads(utts, merged, names, request))


def _color(spk: int) -> str:
    return PALETTE[spk % len(PALETTE)]


def _turn(u, show_speaker=True, continued=False, names=None) -> str:
    span = f"{_fmt(u['start'])} – {_fmt(u['end'])}"
    speaker = (f"<span class='pk-spk' style='--c:{_color(u['speaker'])}' data-spk-k='{u['speaker']}'>"
               f"{html.escape(_name(u['speaker'], names))}</span>" if show_speaker else "")
    return (f"<div class='pk-turn{' pk-cont' if continued else ''}{'' if show_speaker else ' pk-nospk'}' "
            f"data-s='{u['start']:.2f}' data-e='{u['end']:.2f}' title='{span} · click to play'>"
            f"<span class='pk-ts'>{_fmt(u['start'])}</span>{speaker}"
            f"<span class='pk-txt'>{html.escape(u['text'])}</span></div>")


# What the result views say before there is a result.
NOTHING_YET = ("<p class='pk-empty'>Nothing here yet: upload or record a meeting above, then press "
               "<b>2. Transcribe</b> to see who said what. Or <a href='example/'>see it on a real meeting</a>.</p>")
SEGMENTS_HINT = "Click a row to play it in the timeline above."
NO_AUDIO = "Upload or record some audio first."
NO_SPEECH = "No speech found in this recording."
# The GPU is shared with other services; when it is short of memory even after a retry (pipeline.transcribe).
GPU_FULL = "The GPU is short of memory right now. Please try again in a minute; no transcription was used."


class NoSpeech(Exception):
    """The diarizer found nobody speaking."""


def _stop_with_warning(message):
    """End the event with a warning popup only (no error popup or badge): for what the user can fix."""
    gr.Warning(message)
    raise gr.Error(message, visible=False, print_exception=False)


def needs_audio(path, refused=""):
    """Runs before the GPU queue: with nothing to work on, say why (a warning only) and stop there.

    `refused` is why the last file added was turned down, which is what someone looking at that file in
    the upload box needs to hear, rather than "upload something".
    """
    if not path:
        _stop_with_warning(refused or NO_AUDIO)  # also stops the `.success()` that would run the model


def render_conversation(utts, names=None, search=True) -> str:
    """One row per turn: time, speaker, text. A speaker's consecutive turns share one label.

    With `search`, a box above it filters the turns and marks the matches on the timeline (PLAY_ON_CLICK_JS).
    """
    if not utts:
        return "<p class='pk-empty'>No speech recognized.</p>"
    rows = [_turn(u, continued=i > 0 and utts[i - 1]["speaker"] == u["speaker"], names=names)
            for i, u in enumerate(utts)]
    bar = ("<div class='pk-searchbar'><input class='pk-search' type='search' placeholder='Search the conversation…' "
           "aria-label='Search the conversation'><span class='pk-search-count'></span></div>") if search else ""
    return f"{bar}<div class='pk-transcript'>{''.join(rows)}</div>"


def render_per_speaker(utts, num_speakers, names=None, renamable=False, mergeable=False, hidden=()) -> str:
    shown = [s for s in range(num_speakers) if s not in hidden]
    talks = [sum(u["end"] - u["start"] for u in utts if u["speaker"] == s) for s in range(num_speakers)]
    total = sum(talks) or 1.0
    # Who talked how much, at a glance.
    share = "".join(
        f"<div class='pk-share-row'><span class='pk-spk' style='--c:{_color(s)}' data-spk-k='{s}'>"
        f"{html.escape(_name(s, names))}</span><span class='pk-share-bar'><i style='width:{100 * t / total:.1f}%;"
        f"background:{_color(s)}'></i></span><span class='pk-meta'>{100 * t / total:.0f}%"
        f"<span class='pk-share-time'> · {_fmt(t)}</span></span></div>"
        for s, t in enumerate(talks) if s not in hidden)
    tip = ("<div class='pk-meta pk-share-tip'>Tip: ✎ Rename gives a speaker a name everywhere"
           + ("; Merge into… joins a speaker the diarizer split in two." if mergeable else ".") + "</div>")
    undo = "<button class='pk-unmerge'>Undo merges</button>" if hidden else ""
    sections = [f"<div class='pk-share'><div class='pk-share-title'>Speaking time</div>{share}"
                f"{tip if renamable else ''}{undo}</div>"]
    for s in shown:
        own = [u for u in utts if u["speaker"] == s]
        talk = sum(u["end"] - u["start"] for u in own)
        words = sum(len(u["text"].split()) for u in own)
        # The header collapses / expands the speaker's turns (a native <details>: no script involved).
        rename = (f"<button class='pk-rename' data-rename='{s}' title='Rename this speaker'>✎ Rename</button>"
                  if renamable else "")
        # One person split in two by the diarizer: merge them (the page sends it; the server redraws).
        merge = ("<select class='pk-merge' data-from='{0}' title='Merge this speaker into another'>"
                 "<option value=''>Merge into…</option>{1}</select>").format(s, "".join(
                     f"<option value='{j}'>{html.escape(_name(j, names))}</option>" for j in shown if j != s)
                 ) if mergeable and len(shown) > 1 else ""
        head = (f"<summary class='pk-head' title='Click to collapse / expand'><span class='pk-caret'>▾</span>"
                f"<span class='pk-spk' style='--c:{_color(s)}' data-spk-k='{s}'>{html.escape(_name(s, names))}</span>"
                f"{rename}{merge}<span class='pk-meta'>{words} words · {_fmt(talk)} speaking</span></summary>")
        body = "".join(_turn(u, show_speaker=False) for u in own) or "<p class='pk-empty'>Nothing recognized.</p>"
        sections.append(f"<details class='pk-section' open>{head}{body}</details>")
    return f"<div class='pk-transcript'>{''.join(sections)}</div>"


def seek_to_row(utts, evt: gr.SelectData):
    """A table row was clicked: hand its span to the timeline's player (see `seek_box`)."""
    if not utts or evt.index is None:
        return gr.skip()
    u = utts[evt.index[0]]
    return f"{u['start']:.2f},{u['end']:.2f},{time.time()}"  # the timestamp makes repeat clicks a change


def _dump(path, segments):
    with open(path, "w") as f:
        json.dump(segments, f, indent=1)


def _clock(t: float, sep: str) -> str:
    """Seconds -> `HH:MM:SS<sep>mmm`, as subtitles write it (`,` for SRT, `.` for WebVTT)."""
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


EXPORTS = (("transcript.txt", "Text (.txt)"), ("transcript.srt", "Subtitles (.srt)"),
           ("transcript.vtt", "Subtitles (.vtt)"), ("transcript.seglst.json", "Utterances (SegLST)"),
           ("transcript.words.seglst.json", "Word timestamps (SegLST)"))


def _transcript_text(utts, names=None) -> str:
    return "".join(f"[{_fmt(u['start'])}] {_name(u['speaker'], names)}: {u['text']}\n" for u in utts)


def _export_files(utts, words, names, out_dir) -> dict:
    """Write the result in every download format into `out_dir`: {file name: path}."""
    files = {}

    def write(name, text):
        files[name] = os.path.join(out_dir, name)
        with open(files[name], "w", encoding="utf-8") as f:
            f.write(text)

    write("transcript.txt", _transcript_text(utts, names))
    write("transcript.srt", "".join(
        f"{i}\n{_clock(u['start'], ',')} --> {_clock(u['end'], ',')}\n{_name(u['speaker'], names)}: {u['text']}\n\n"
        for i, u in enumerate(utts, 1)))
    write("transcript.vtt", "WEBVTT\n\n" + "".join(
        f"{_clock(u['start'], '.')} --> {_clock(u['end'], '.')}\n"
        f"<v {_name(u['speaker'], names).replace('>', '').replace('<', '')}>{u['text']}\n\n" for u in utts))
    write("transcript.seglst.json", json.dumps(
        [{"session_id": "audio", "speaker": _name(u["speaker"], names), "start_time": round(u["start"], 2),
          "end_time": round(u["end"], 2), "words": u["text"]} for u in utts], indent=1))
    write("transcript.words.seglst.json", json.dumps(
        [{"session_id": "audio", "speaker": _name(sp, names), "start_time": round(w.start, 2),
          "end_time": round(w.end, 2), "words": w.text} for sp, ws in enumerate(words) for w in ws], indent=1))
    return files


def _downloads_html(urls, copy_text: str) -> str:
    """The download links (`urls`: [(label, url)]) and a button that copies the plain-text transcript."""
    links = "".join(f"<a class='pk-btn pk-btn-2' href='{html.escape(url, quote=True)}' download>{html.escape(label)}</a>"
                    for label, url in urls)
    return (f"<div class='pk-downloads'><span class='pk-dl-label'>Download</span>{links}"
            f"<button class='pk-btn pk-copy' data-copy='{html.escape(copy_text, quote=True)}'>Copy transcript</button></div>")


DOWNLOADS_JS = r"""
element.addEventListener('click', (e) => {
  const button = e.target.closest('[data-copy]');
  if (!button) return;
  navigator.clipboard.writeText(button.dataset.copy).then(() => {
    button.textContent = 'Copied ✓';
    setTimeout(() => { button.textContent = 'Copy transcript'; }, 1800);
  });
});
"""


SEEK_JS = """(v) => {
    if (v && window.pkTimeline) { const [s, e] = v.split(',').map(Number); window.pkTimeline.play(s, e); }
    return [];
}"""

# A recording becomes the source only once its upload has finished and `new_source` has checked it, so a
# button clicked mid-upload would answer "upload some audio first". Instead the click waits: uploads are
# counted at `fetch` (the Gradio client's upload goes through it), and the wait lasts until `new_source`
# has run for the last one (SOURCE_SETTLED_JS).
UPLOAD_WAIT_JS = r"""
if (!window.pkUploads) {
  window.pkUploads = 0;
  window.pkAwaitSource = false;
  const fetch0 = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (!/\/upload(\?|$)/.test(url)) return fetch0.apply(this, arguments);
    window.pkUploads++;
    window.pkAwaitSource = true;
    window.pkUploadEnded = 0;
    const done = (ok) => {
      window.pkUploads--;
      window.pkUploadEnded = Date.now();
      if (!ok && !window.pkUploads) window.pkAwaitSource = false;  // nothing will arrive to check
    };
    return fetch0.apply(this, arguments).then((r) => { done(r.ok); return r; },
                                              (e) => { done(false); throw e; });
  };
  window.pkSourceReady = async (button) => {
    if (!window.pkUploads && !window.pkAwaitSource) return;
    const label = button && button.textContent;
    if (button) button.textContent = 'Waiting for the upload…';
    // Generous after the upload itself ends: the check of the file is quick, but it is queued.
    while (window.pkUploads || (window.pkAwaitSource && Date.now() - window.pkUploadEnded < 30000))
      await new Promise((r) => setTimeout(r, 200));
    if (button) button.textContent = label;
  };
}
"""
SOURCE_SETTLED_JS = "() => { window.pkAwaitSource = false; }"


def _wait_for_upload_js(button_id: str) -> str:
    return f"async (...args) => {{ await window.pkSourceReady?.(document.querySelector('#{button_id}')); return args; }}"

# Named on both pages: where the diarization priors come from.
DIARIZER_LINK = ("<a href='https://huggingface.co/nvidia/Nemotron-3-Diarization' target='_blank' rel='noopener'>"
                 "NVIDIA Nemotron-3-Diarization</a>")
# Short form, for the tables' column headers (where "+ DiariZen" is a link too).
NEMOTRON_LINK = ("<a href='https://huggingface.co/nvidia/Nemotron-3-Diarization' target='_blank' rel='noopener'>"
                 "Nemotron-3-<wbr>Diarization</a>")
MODEL_CARD = "https://huggingface.co/BUT-FIT/Pandemonium"
MODEL_LINK = f"<a href='{MODEL_CARD}' target='_blank' rel='noopener'>Pandemonium</a>"

with gr.Blocks(title="Pandemonium preview") as demo:
    gr.Markdown(
        "# Pandemonium <small>· preview</small>\n"
        "Who said what, in meetings up to " + LIMIT + " long. Upload a recording: Pandemonium finds the speakers "
        "and transcribes every one of them, with timestamps, in a single pass. Diarization priors come from "
        + DIARIZER_LINK + ". <a href='" + MODEL_CARD + "' target='_blank' rel='noopener'>Model card</a> · "
        "<a href='example/'>See it on a real meeting</a>."
    )
    gr.HTML(f"<div class='pk-notice'><b>English only.</b> Recordings up to <b>{LIMIT}</b>, with up to "
            "<b>8 speakers</b>. Other languages are not supported and will produce wrong output.</div>",
            padding=False)
    account = gr.HTML(padding=False)
    demo.load(account_status, None, account, show_progress="hidden")
    diar_state = gr.State(None)
    source = gr.State(None)  # path of the recording in use: the upload or the microphone take

    with gr.Tabs():
        with gr.Tab("Upload"):
            file_in = gr.File(label=f"English audio recording, up to {LIMIT}", type="filepath", elem_id="pk-upload",
                              file_types=["audio", *ACCEPTED_EXTENSIONS], height=140)
        with gr.Tab("Record"):
            mic_in = gr.Audio(sources=["microphone"], type="filepath", label=f"Record in English (up to {LIMIT})",
                              buttons=[])  # no download / share: audio is never offered for download
    source_note = gr.HTML(padding=False, js_on_load=UPLOAD_WAIT_JS)  # why the file just added was refused, if it was
    refused_state = gr.State("")
    with gr.Row():
        diar_btn = gr.Button("1. Find speakers", variant="secondary", elem_id="pk-diar-btn")
        asr_btn = gr.Button("2. Transcribe", variant="primary", elem_id="pk-asr-btn")
    # Shown when Transcribe found an earlier result for this recording (see `_results`).
    with gr.Row(visible=False, elem_classes="pk-again") as again_row:
        again_note = gr.HTML(padding=False, min_width=280)
        again_btn = gr.Button("Transcribe afresh (uses 1 transcription)", variant="secondary", size="sm", scale=0,
                              min_width=290)
    names_box = gr.Textbox(visible="hidden", elem_id="pk-names")  # speaker renames, set by the page (pkRename)
    names_apply = gr.Button(visible="hidden", elem_id="pk-names-apply")  # ... which then clicks this to send them
    merges_box = gr.Textbox(visible="hidden", elem_id="pk-merges")  # merged speakers, set by the page (pkMerge)
    merges_apply = gr.Button(visible="hidden", elem_id="pk-merges-apply")
    words_state = gr.State(None)
    diar_info = gr.Markdown()
    diar_plot = gr.HTML(js_on_load=TIMELINE_JS, padding=False)
    # Carries a clicked table row's span to the timeline's player, client-side.
    seek_box = gr.Textbox(visible="hidden")

    utts_state = gr.State([])
    with gr.Tabs():
        with gr.Tab("Conversation"):
            conversation = gr.HTML(NOTHING_YET, js_on_load=PLAY_ON_CLICK_JS)
        with gr.Tab("Per speaker"):
            per_speaker = gr.HTML(NOTHING_YET, js_on_load=PLAY_ON_CLICK_JS)
        with gr.Tab("Segments"):
            segments_hint = gr.HTML(NOTHING_YET)
            table = gr.Dataframe(visible=False, headers=["Start", "End", "Speaker", "Text"], datatype=["str", "str", "str", "str"],
                                 interactive=False, wrap=True, show_search="search", max_height=520,
                                 column_widths=["9%", "9%", "12%", "70%"])
    downloads = gr.HTML(visible=False, padding=False, js_on_load=DOWNLOADS_JS)

    gr.Markdown(
        "<small>Research preview: built for English meeting recordings with up to 8 speakers; other languages "
        "and domains may be unreliable. Please don't upload anything confidential. "
        "Contact: <a href='mailto:ipoloka@fit.vut.cz'>ipoloka@fit.vut.cz</a>, BUT Speech@FIT · "
        "<a href='privacy'>Privacy notice</a> · <a href='delete'>Delete my account</a></small>"
    )

    def new_source(path, request: gr.Request):
        """A new upload or recording: check it, use it, and clear everything derived from the last one.

        Checked here, from the header, so a video or an over-long file is refused the moment it is
        added rather than when a button is pressed; a refused file does not become the source.
        """
        hidden = gr.update(visible=False)
        refused = ""
        _own(path, request.username)  # also covers a microphone take that Gradio re-saved under a new path
        if path:
            try:
                seconds = _probe_seconds(path)
                if seconds > MAX_SECONDS + 0.5:
                    raise _too_long(seconds)
            except gr.Error as err:
                gr.Warning(err.message)
                refused = err.message
                path = None
        # A refused file stays in the upload box, so say so next to it too, not only in a passing popup.
        note = f"<div class='pk-refused'>{html.escape(refused)}</div>" if path is None and refused else ""
        if path:
            done = _earlier(request.username, path)
            note = (f"<div class='pk-ready'><b>{html.escape(os.path.basename(path))}</b> · {_fmt(seconds)} · ready. "
                    + ("You have transcribed this recording before: <b>2. Transcribe</b> shows that transcript "
                       "again, without using a transcription."
                       if done else "<b>2. Transcribe</b> usually takes under a minute; <b>1. Find speakers</b> "
                       "first shows who speaks when.") + "</div>")
        return (path, refused, note, None, "", "", NOTHING_YET, NOTHING_YET, gr.update(value=None, visible=False),
                NOTHING_YET, [], None, hidden, hidden, "", "", *idle())

    reset_outputs = [source, refused_state, source_note, diar_state, diar_plot, diar_info, conversation, per_speaker,
                     table, segments_hint, utts_state, words_state, downloads, again_row, names_box, merges_box]
    # Only the timeline shows the busy state: Gradio's default overlays every output at once, which
    # makes the whole page look broken mid-run.
    # Both buttons queue for the one GPU together (the pipeline also locks it, for API callers).
    # A click during an upload first waits for it (UPLOAD_WAIT_JS).
    # `needs_audio` answers a click with nothing uploaded at once: it is queued (warnings travel through the
    # queue) but in its own unlimited group, so it never waits behind a GPU job.
    # While a job runs, its buttons are disabled: a second click would queue (and charge) a second run.
    buttons = [diar_btn, asr_btn, again_btn]
    busy = lambda: [gr.update(interactive=False)] * 3  # noqa: E731
    idle = lambda: [gr.update(interactive=True)] * 3  # noqa: E731
    results = [diar_state, diar_plot, conversation, per_speaker, table, segments_hint, utts_state, words_state,
               downloads, account, again_row, again_note, merges_box]

    def gpu_job(start, fn, inputs, outputs):
        """`fn` on the GPU queue, the buttons disabled meanwhile. Re-enabled on failure too: in this Gradio
        `.then` does not run after an error (despite its docs), which left them dead until a reload."""
        job = start.then(fn, inputs, outputs, show_progress_on=diar_plot, concurrency_id="gpu",
                         concurrency_limit=gpus.size)
        job.then(idle, None, buttons, queue=False)
        job.failure(idle, None, buttons, queue=False)
        return job

    diar_click = diar_btn.click(needs_audio, [source, refused_state], None, concurrency_limit=None,
                                show_progress="hidden", js=_wait_for_upload_js("pk-diar-btn"))
    asr_click = asr_btn.click(needs_audio, [source, refused_state], None, concurrency_limit=None,
                              show_progress="hidden", js=_wait_for_upload_js("pk-asr-btn"))
    jobs = [
        gpu_job(diar_click.success(busy, None, buttons, queue=False),
                run_diarization, [source, names_box], [diar_state, diar_plot, diar_info]),
        gpu_job(asr_click.success(busy, None, buttons, queue=False),
                run_transcription, [source, diar_state, names_box], results),
        gpu_job(again_btn.click(busy, None, buttons, queue=False),
                run_transcription_again, [source, diar_state, names_box], results),
    ]
    # A new (or removed) recording cancels this page's job still queued or running for the old one: its
    # result would land on a page that no longer shows that recording.
    for src in (file_in, mic_in):
        settled = src.change(new_source, src, reset_outputs + buttons, show_progress="hidden", cancels=jobs)
        settled.then(None, js=SOURCE_SETTLED_JS)
        settled.failure(None, js=SOURCE_SETTLED_JS)
    names_apply.click(apply_names, [names_box, merges_box, words_state, diar_state], [table, downloads],
                      show_progress="hidden")
    merges_apply.click(apply_merges, [merges_box, names_box, source, words_state, diar_state],
                       [diar_plot, conversation, per_speaker, table, utts_state, downloads], show_progress="hidden")
    table.select(seek_to_row, utts_state, seek_box, show_progress="hidden")
    seek_box.change(None, seek_box, None, show_progress="hidden", js=SEEK_JS)


# ---------------------------------------------------------------------------------------------
# The public example: a finished result anyone can explore before asking for an access code.
# Everything is precomputed (example/README.md); nothing runs and nothing counts against a quota.
# ---------------------------------------------------------------------------------------------

EXAMPLE = load_example(os.environ.get("DEMO_EXAMPLE", "ami_IS1009a"), decode=_decode)
# The example's files are served by name from `example/media/<name>` (see `__main__`), not through Gradio's
# file route, whose URLs carry the absolute path on this machine.
# Outside Gradio's temp folder (served by /example/media, not by Gradio), so the weekly clean-up leaves them.
EXAMPLE_EXPORTS = _export_files(EXAMPLE.utts, EXAMPLE.words, None, tempfile.mkdtemp(prefix="example-exports-"))
EXAMPLE_MEDIA = {"audio.mp3": EXAMPLE.audio_path, **EXAMPLE_EXPORTS}
if "reference" in EXAMPLE.downloads:
    EXAMPLE_MEDIA["reference.seglst.json"] = EXAMPLE.downloads["reference"]
# Where the example opens: `start_s` in its meta.json, else where the human reference begins (the
# meeting itself; AMI does not transcribe the chatter before it), else the beginning.
EXAMPLE_START = float(EXAMPLE.meta.get("start_s", round(EXAMPLE.reference[0]["start"]) if EXAMPLE.reference else 0))


def _example_about(ex) -> str:
    """What the recording is, how well the model did on it, and the attribution its licence requires."""
    meta, scores, src = ex.meta, ex.scores or {}, ex.meta.get("source", {})
    corpus = html.escape(src.get("corpus", "the reference"))
    score = ""
    if "cpWER" in scores:
        score = (f" Against {corpus}'s human transcript: <b>cpWER {scores['cpWER']:.2f}%</b>, "
                 f"tcpWER {scores['tcpWER_collar5s']:.2f}% (5 s collar), {scores['reference_words']:,} reference words.")
    credit = ""
    if src.get("attribution"):
        lic = (f" <a href='{html.escape(src['license_url'])}' target='_blank'>{html.escape(src['license'])}</a>"
               if src.get("license_url") else "")
        url = f" · <a href='{html.escape(src['url'])}' target='_blank'>corpus</a>" if src.get("url") else ""
        credit = f"<div class='pk-meta'>Source: {html.escape(src['attribution'])}{lic}{url}</div>"
    return (f"<div class='pk-about'><div><b>{html.escape(meta['title'])}</b>. {html.escape(meta.get('description', ''))}"
            f"{score}</div>{credit}</div>")


# tcpWER (%, 5 s collar) on each set's full sessions: (dataset, where to get it, this demo, SE-DiCoW + DiariZen).
# A name ending in "*", "†", "‡" or "§" refers to that line of BENCHMARK_FOOTNOTES. The first BENCHMARK_IN_DOMAIN sets
# are in Pandemonium's training data; the rest are out of domain for it, and the table says so (the diarizer
# that supplies its priors is another matter: "†").
BENCHMARK_IN_DOMAIN = 2
BENCHMARK_RESULTS = [
    ("NOTSOFAR\u20111", "https://huggingface.co/datasets/microsoft/NOTSOFAR", 18.23, 25.83),  # non-breaking hyphen
    ("AMI", "https://huggingface.co/datasets/edinburghcstr/ami", 16.76, 19.28),
    ("ICSI†", "https://groups.inf.ed.ac.uk/ami/icsi/", 14.82, 20.73),
    ("Mixer 6‡", "https://catalog.ldc.upenn.edu/LDC2013S03", 12.28, 12.20),
    ("Santa Barbara Corpus*", "https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english",
     19.09, 21.58),
    ("DiPCo†§", "https://zenodo.org/records/8122551", 24.90, 35.98),
]
# Real-time factor, diarization + ASR, on the median-length recording of each set (see the note below the table).
BENCHMARK_RTFX = (426, 28.9)
# The SBC evaluation set (evals/sbcsae-sdm_eval.tar, as in the paper) is a subset of the corpus.
BENCHMARK_FOOTNOTES = [
    "* Not the full corpus: 44 of its 60 recordings (17.3 h), with 2–7 speakers each.",
    "† Out of domain for Pandemonium only: Nemotron-3-Diarization, which supplies the diarization priors, was "
    "trained on all of ICSI and on the DiPCo dev set.",
    "‡ Evaluated on the UEM-split sets; microphone CH08.",
    "§ Microphone U04 CH4 (array 4, mic 4): channel 24 of the 35-channel recording.",
]
# Single-speaker English, the Open ASR Leaderboard sets (all 82k clips), WER % with the leaderboard's
# normalizer: (set, Parakeet-TDT-0.6b-v3, Pandemonium + Nemotron as in this demo).
SINGLE_SPEAKER_RESULTS = [
    ("LibriSpeech clean", 1.94, 2.35),
    ("LibriSpeech other", 3.59, 5.00),
    ("VoxPopuli", 6.10, 15.26),
    ("Earnings-22", 11.18, 16.07),
    ("AMI (close-talk)", 11.34, 14.13),
    ("GigaSpeech", 9.53, 10.89),
    ("SPGISpeech", 3.98, 5.54),
]
SINGLE_SPEAKER_MEAN = (6.81, 9.89)
PARAKEET = ("<a href='https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3' target='_blank' rel='noopener'>"
            "Parakeet-TDT-0.6b-v3</a>")
SE_DICOW = "<a href='https://huggingface.co/BUT-FIT/SE-DiCoW' target='_blank' rel='noopener'>SE-DiCoW</a>"
DIARIZEN = ("<a href='https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2' target='_blank' "
            "rel='noopener'>DiariZen</a>")


def _benchmarks() -> str:
    def pair(ours, theirs, fmt, lower_is_better=True):
        best = min(ours, theirs) if lower_is_better else max(ours, theirs)
        return "".join(f"<td{' class=pk-best' if v == best else ''}>{fmt(v)}</td>" for v in (ours, theirs))

    wer = lambda v: f"{v:.2f}"
    def group(cls, label, note):
        return (f"<tr class='pk-group {cls}'><td colspan='3'><span class='pk-tag'>{label}</span>"
                f"<small>{note}</small></td></tr>")

    def row(name, url, ours, theirs, cls=""):
        return (f"<tr{cls}><td><a href='{html.escape(url)}' target='_blank' rel='noopener'>{html.escape(name)}</a>"
                f"</td>{pair(ours, theirs, wer)}</tr>")

    rows = group("pk-in", "In domain", "Pandemonium trained on their train splits only; scored on the held-out "
                                      "eval sets")
    rows += "".join(row(*r) for r in BENCHMARK_RESULTS[:BENCHMARK_IN_DOMAIN])
    rows += group("pk-ood", "Out of domain", "never seen by Pandemonium in training")
    rows += "".join(row(*r, cls=" class='pk-ood-row'") for r in BENCHMARK_RESULTS[BENCHMARK_IN_DOMAIN:])
    n = len(BENCHMARK_RESULTS)
    mean = (sum(r[2] for r in BENCHMARK_RESULTS) / n, sum(r[3] for r in BENCHMARK_RESULTS) / n)
    rows += f"<tr class='pk-sum'><td>Mean</td>{pair(*mean, wer)}</tr>"
    rows += (f"<tr class='pk-sum'><td>Speed (RTFx)</td>"
             f"{pair(*BENCHMARK_RTFX, lambda v: f'{v:g}×', lower_is_better=False)}</tr>")
    return ("<div class='pk-bench'><div class='pk-bench-title'>Comparison on public meeting benchmarks</div>"
            "<table><thead><tr><th>Dataset · tcpWER %</th><th>" + MODEL_LINK + "<br><small>+ " + NEMOTRON_LINK + "</small></th>"
            f"<th>{SE_DICOW}<br><small>+ {DIARIZEN}</small></th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            f"<tfoot><tr><td colspan='3'>{'<br>'.join(map(html.escape, BENCHMARK_FOOTNOTES))}</td></tr></tfoot></table>"
            "<div class='pk-meta'><b>tcpWER</b>: word error rate in which each word must be attributed to the right "
            "speaker and fall within 5 s of its reference time (5 s collar); lower is better. Full sessions of "
            "each set (Mixer 6: ‡), one distant microphone: the standard single-channel track of each set unless "
            "marked, text normalized with the CHiME-8 text normalization. Pandemonium: this demo's pipeline (diarization priors from "
            + DIARIZER_LINK + "). SE-DiCoW: greedy decoding, self-enrolled, diarized by DiariZen.</div>"
            "<div class='pk-meta'><b>RTFx</b>: audio duration ÷ (diarization + transcription time); higher is "
            "faster. Measured on the median-length recording of each set (6 recordings, 167 min in total, 2–7 "
            "speakers, 6–57 min each), each system alone on the same GPU (RTX PRO 6000 Blackwell) with 8 CPU "
            "threads; model loading and a warm-up run excluded.</div>" + _single_speaker() + "</div>")


def _single_speaker() -> str:
    """The cost on single-speaker speech: Pandemonium against the ASR model it starts from."""
    def cells(values):
        best = min(values)
        return "".join(f"<td{' class=pk-best' if v == best else ''}>{v:.2f}</td>" for v in values)

    rows = "".join(f"<tr><td>{html.escape(name)}</td>{cells(v)}</tr>" for name, *v in SINGLE_SPEAKER_RESULTS)
    rows += f"<tr class='pk-sum'><td>Mean</td>{cells(SINGLE_SPEAKER_MEAN)}</tr>"
    return ("<div class='pk-bench-title pk-bench-sub'>Single-speaker English</div>"
            "<div class='pk-meta'>Built for meetings, Pandemonium trails the model it starts from on single-speaker "
            "speech.</div>"
            "<div class='pk-scroll'><table class='pk-single'><thead>"
            "<tr><th>Dataset · WER %</th>"
            f"<th>{PARAKEET.replace('TDT-', 'TDT-<wbr>')}<br><small>base model</small></th>"
            f"<th>{MODEL_LINK}<br><small>+ {NEMOTRON_LINK}</small></th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            "<div class='pk-meta'>The <a href='https://huggingface.co/spaces/hf-audio/open_asr_leaderboard' "
            "target='_blank' rel='noopener'>Open ASR Leaderboard</a> test sets, all 82k clips, text normalized "
            "with the leaderboard's normalizer. Pandemonium: this demo's pipeline, diarization included.</div>")


BENCHMARKS = _benchmarks()

# What the timeline's marks mean (the example page, where first-time visitors meet it).
TIMELINE_LEGEND = (
    "<div class='pk-legend'><span><i class='pk-lg-block'></i>a turn, with its text (click to play)</span>"
    "<span><i class='pk-lg-true'></i>how long the turn really lasts</span>"
    "<span><i class='pk-lg-prob'></i>how likely the speaker is talking</span>"
    "<span><i class='pk-lg-ref'></i>human reference (toggle it in the toolbar)</span></div>"
)


def _call_to_action(text: str) -> str:
    return ("<div class='pk-cta'>"
            f"<div>{text}</div>"
            "<div class='pk-cta-actions'><a class='pk-btn' href='../signup'>Request an access code</a>"
            "<a class='pk-btn pk-btn-2' href='../?login'>Log in</a></div></div>")


# Above the example, introducing it; and again at the end of the page, for whoever has seen enough.
EXAMPLE_CALL_TO_ACTION = _call_to_action(
    "<b>See it on a real meeting.</b> Below is Pandemonium's output for a recorded meeting: play it, click "
    "any turn to hear it, compare it with the human transcript. No account needed. To transcribe your own "
    f"recordings, request a free access code ({access.DEFAULT_QUOTA} transcriptions).")
EXAMPLE_CALL_TO_ACTION_END = _call_to_action(
    "<b>Try it on your own recordings.</b> Request a free access code "
    f"({access.DEFAULT_QUOTA} transcriptions), or log in if you already have one.")

with gr.Blocks(title="Pandemonium preview: example") as explore:
    gr.Markdown(
        "# Pandemonium <small>· preview</small>\n"
        "Who said what, in a meeting recording. Pandemonium finds the speakers of a recording "
        "and transcribes every one of them, with timestamps, in a single pass. Diarization priors come from "
        + DIARIZER_LINK + ". <a href='" + MODEL_CARD + "' target='_blank' rel='noopener'>Model card</a>."
    )
    gr.HTML(EXAMPLE_CALL_TO_ACTION, padding=False)
    gr.HTML(BENCHMARKS, padding=False)  # the headline numbers
    gr.HTML(_example_about(EXAMPLE), padding=False)
    gr.HTML(TIMELINE_LEGEND, padding=False)
    gr.HTML(timeline(EXAMPLE.audio_path, EXAMPLE.audio, EXAMPLE.diar, EXAMPLE.utts, open_at=EXAMPLE_START,
                     reference=EXAMPLE.reference, src="media/audio.mp3", renamable=True),
            js_on_load=TIMELINE_JS,
            padding=False)
    ex_seek = gr.Textbox(visible="hidden")
    ex_utts = gr.State(EXAMPLE.utts)
    with gr.Tabs():
        with gr.Tab("Conversation"):
            gr.HTML(render_conversation(EXAMPLE.utts), js_on_load=PLAY_ON_CLICK_JS)
        with gr.Tab("Per speaker"):
            gr.HTML(render_per_speaker(EXAMPLE.utts, EXAMPLE.diar.num_speakers, renamable=True),
                    js_on_load=PLAY_ON_CLICK_JS)
        with gr.Tab("Segments"):
            gr.Markdown("Click a row to play it in the timeline above.")
            ex_table = gr.Dataframe(
                value=[[_fmt(u["start"]), _fmt(u["end"]), _name(u["speaker"]), u["text"]] for u in EXAMPLE.utts],
                headers=["Start", "End", "Speaker", "Text"], datatype=["str", "str", "str", "str"],
                interactive=False, wrap=True, show_search="search", max_height=520,
                column_widths=["9%", "9%", "12%", "70%"])
        if EXAMPLE.reference:
            with gr.Tab("Human reference"):
                mapping = ", ".join(f"{spk} = {name}" for spk, name in sorted(
                    EXAMPLE.scores["speaker_mapping"].items(), key=lambda kv: kv[1]))
                late = EXAMPLE.reference[0]["start"] - EXAMPLE.utts[0]["start"] > 5
                gr.Markdown(f"The corpus's own transcript, its participants shown as the speakers Pandemonium "
                            f"found ({mapping}). Click a turn to play it."
                            + (f" It starts at {_fmt(EXAMPLE.reference[0]['start'])}, when the meeting does; what is "
                               "said before that is not in it." if late else ""))
                gr.HTML(render_conversation(EXAMPLE.reference, search=False), js_on_load=PLAY_ON_CLICK_JS)
    gr.HTML(_downloads_html([(label, f"media/{name}") for name, label in EXPORTS]
                            + ([("Human reference (SegLST)", "media/reference.seglst.json")]
                               if "reference.seglst.json" in EXAMPLE_MEDIA else []),
                            _transcript_text(EXAMPLE.utts)), padding=False, js_on_load=DOWNLOADS_JS)
    gr.HTML(EXAMPLE_CALL_TO_ACTION_END, padding=False)
    gr.Markdown(
        "<small>Research preview: built for English meeting recordings with up to 8 speakers. "
        "Contact: <a href='mailto:ipoloka@fit.vut.cz'>ipoloka@fit.vut.cz</a>, BUT Speech@FIT · "
        "<a href='../privacy'>Privacy notice</a></small>"
    )
    ex_table.select(seek_to_row, ex_utts, ex_seek, show_progress="hidden")
    ex_seek.change(None, ex_seek, None, show_progress="hidden", js=SEEK_JS)


def _access_code():
    """The access code: `DEMO_ACCESS_CODE`, else the first line of `DEMO_ACCESS_CODE_FILE`."""
    code = os.environ.get("DEMO_ACCESS_CODE", "").strip()
    path = os.environ.get("DEMO_ACCESS_CODE_FILE")
    if not code and path and os.path.exists(path):
        with open(path) as f:
            code = f.readline().strip()
    return code


def _check(username, password):
    # Personal email + emailed code, or `admin` + the master code. Constant-time comparisons.
    return access.check_login(username, password, ACCESS_CODE)


def _add_signup_routes(app):
    """`/signup`, `/privacy`: public, on the server the two Gradio apps are mounted in."""
    from fastapi import Request
    from fastapi.responses import HTMLResponse

    async def form(request: Request):
        return HTMLResponse(access.signup_page())

    async def submit(request: Request):
        f = await request.form()
        ip = access.client_ip(request.headers, request.client.host if request.client else "")
        done, message = await asyncio.to_thread(  # sends email: keep it off the event loop
            access.signup, f.get("name", ""), f.get("email", ""), f.get("affiliation", ""),
            f.get("purpose", ""), bool(f.get("agree")), ip)
        return HTMLResponse(access.signup_done_page(message) if done else access.signup_page(message, dict(f)))

    async def privacy(request: Request):
        return HTMLResponse(access.privacy_page())

    app.add_api_route("/signup", form, methods=["GET"])
    app.add_api_route("/signup", submit, methods=["POST"])
    app.add_api_route("/privacy", privacy, methods=["GET"])

    async def delete_form(request: Request):
        return HTMLResponse(access.delete_page())

    async def delete_submit(request: Request):
        f = await request.form()
        message = await asyncio.to_thread(  # may send email
            access.request_deletion, f.get("email", ""), f.get("code", ""), _client(request))
        return HTMLResponse(access.delete_page(message, f.get("email", "")))

    async def delete_confirm_form(request: Request):
        token = request.query_params.get("token", "")
        return HTMLResponse(access.delete_confirm_page(token, access.deletion_email(token)))

    async def delete_confirm(request: Request):
        account = access.confirm_deletion((await request.form()).get("token", ""))
        if account:
            _purge_account(account)
        response = HTMLResponse(access.delete_done_page(bool(account)))
        if account:  # the browser's login cookies belong to an account that is gone
            for name in request.cookies:
                if name.startswith("access-token"):
                    response.delete_cookie(name, path="/")
        return response

    # Deleting is only ever done by the button (POST): mail scanners open links (GET) on their own.
    app.add_api_route("/delete", delete_form, methods=["GET"])
    app.add_api_route("/delete", delete_submit, methods=["POST"])
    app.add_api_route("/delete/confirm", delete_confirm_form, methods=["GET"])
    app.add_api_route("/delete/confirm", delete_confirm, methods=["POST"])


# Login sessions live in the server's memory, so a restart logs everyone out and an open page only
# finds out when its next request comes back 401 (an upload, a button). Say so and go to the login
# form, instead of leaving a raw error. Wraps `fetch`, through which every Gradio API call goes.
SESSION_EXPIRED_JS = """() => {
  if (window.pkSessionWatch) return;
  window.pkSessionWatch = true;
  const fetch0 = window.fetch.bind(window);
  let shown = false;
  const expired = () => {
    shown = true;
    const box = document.createElement('div');
    box.className = 'pk-expired';
    box.innerHTML = `<div class="pk-expired-card"><b>Your session has expired.</b>
      <p>The preview was restarted or you were logged out. Please log in again with your email address and
      access code.</p>
      <a class="pk-btn" href="./?login">Log in again</a>
      <p class="pk-meta">Taking you to the login page in <span>8</span> s</p></div>`;
    document.body.appendChild(box);
    let n = 8;
    const left = box.querySelector('span');
    setInterval(() => { if (--n <= 0) location.href = './?login'; else left.textContent = n; }, 1000);
  };
  window.fetch = async (...args) => {
    const response = await fetch0(...args);
    const url = String((args[0] && args[0].url) || args[0]);
    if (response.status === 401 && !shown && url.includes('gradio_api/')) expired();
    return response;
  };
}"""


# What the public example app may answer under `gradio_api/`: the queue for its one event (a table row
# plays its turn) and the heartbeat. Not uploads, file serving, or the API description.
EXAMPLE_API = ("/example/gradio_api/queue/join", "/example/gradio_api/queue/data", "/example/gradio_api/heartbeat")


def _hidden(path: str) -> bool:
    """Routes nobody needs: the public app's API beyond EXAMPLE_API, and the API descriptions of both apps."""
    if path.startswith("/example/gradio_api/") and not path.startswith(EXAMPLE_API):
        return True
    return path.endswith(("/gradio_api/info", "/openapi.json", "/docs", "/redoc")) or path in (
        "/gradio_api/info", "/openapi.json")


# Sent with every response: never framed by another site (clickjacking), no content-type guessing, and
# no full URLs leaking to other sites in the Referer header.
SECURITY_HEADERS = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
                    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin"}

# Failed logins per network: a few typos are fine, a script guessing codes is stopped for a while.
LOGIN_FAILURES, LOGIN_WINDOW = 10, 15 * 60
_login_failures: dict = {}


def _client(request) -> str:
    return access.client_ip(request.headers, request.client.host if request.client else "")


def _login_blocked(request) -> bool:
    now = time.time()
    recent = [t for t in _login_failures.get(_client(request), []) if t > now - LOGIN_WINDOW]
    _login_failures[_client(request)] = recent
    return len(recent) >= LOGIN_FAILURES


def _login_failed(request):
    _login_failures.setdefault(_client(request), []).append(time.time())


# Whether a session's account is still allowed in, looked up at most every 30 s per account.
_active_cache: dict = {}


def _still_active(user) -> bool:
    now, cached = time.time(), _active_cache.get(user)
    if cached and cached[0] > now - 30:
        return cached[1]
    active = access.is_active(user)
    _active_cache[user] = (now, active)
    return active


UPLOAD_MAX_AGE_DAYS = 7


def _clean_uploads():
    """Delete uploads and results older than a week, every 6 hours (also kept from a restart by PrivateTmp)."""
    root = os.environ["GRADIO_TEMP_DIR"]
    while True:
        cutoff = time.time() - UPLOAD_MAX_AGE_DAYS * 86400
        for folder, dirs, files in os.walk(root, topdown=False):
            for name in files:
                path = os.path.join(folder, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        _owners.pop(os.path.realpath(path), None)
                except OSError:
                    pass
            if folder not in (root, RESULTS_DIR) and not os.listdir(folder):
                try:
                    os.rmdir(folder)
                except OSError:
                    pass
        time.sleep(6 * 3600)


# The card shown when the demo's link is shared (static/make_preview.py draws it).
PREVIEW_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "preview.png")
PREVIEW_TITLE = "Pandemonium · who said what, in meetings"
PREVIEW_TEXT = ("Upload a meeting recording: Pandemonium finds the speakers and transcribes every one of them, "
                "with timestamps. A research preview by BUT Speech@FIT.")


def _preview_meta() -> str:
    """Open Graph / Twitter tags: title, description and, with a public URL to point at, the image."""
    url = access.PUBLIC_URL
    tags = [("property", "og:type", "website"), ("property", "og:site_name", "Pandemonium"),
            ("property", "og:title", PREVIEW_TITLE), ("property", "og:description", PREVIEW_TEXT),
            ("name", "description", PREVIEW_TEXT), ("name", "twitter:card", "summary_large_image"),
            ("name", "twitter:title", PREVIEW_TITLE), ("name", "twitter:description", PREVIEW_TEXT)]
    if url:
        image = url.rstrip("/") + "/preview.png"
        tags += [("property", "og:url", url), ("property", "og:image", image), ("property", "og:image:width", "1200"),
                 ("property", "og:image:height", "630"), ("property", "og:image:alt", PREVIEW_TITLE),
                 ("name", "twitter:image", image)]
    return "".join(f'<meta {kind}="{key}" content="{html.escape(value, quote=True)}" />' for kind, key, value in tags)


class _WithPrefix:
    """Put the proxy's stripped prefix back on a mounted Gradio app's request path.

    Gradio needs `root_path` (`/pandemonium/example`) to build its public URLs, and Starlette then only
    routes requests whose path starts with it. nginx strips `/pandemonium`, so `/example/config` would
    404; restoring the prefix makes the path and the root path agree again. The app at "/" needs none
    of this: no path starts with `/pandemonium`, and Starlette then routes it unchanged.
    """

    def __init__(self, app, prefix: str):
        self.app, self.prefix = app, prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = self.prefix + scope["path"]
            scope = dict(scope, path=path, raw_path=path.encode())
        await self.app(scope, receive, send)


ACCESS_CODE = _access_code()  # the admin's master code

# Gradio labels its login fields "username" / "password" (translated into the browser's language), and
# neither its translations nor `head` reach the login page, so the server adds this to that page: it
# names the fields what they are everywhere else, whenever they are (re)rendered.
LOGIN_LABELS_JS = """<script>
(() => {
  const name = (input, text) => {
    const span = input && input.closest('label') && input.closest('label').querySelector('span');
    if (span && span.textContent.trim() !== text) span.textContent = text;
  };
  new MutationObserver(() => {
    const pw = document.querySelector('input[type=password]');
    if (!pw) return;
    const inputs = [...document.querySelectorAll('input[type=text], input[type=password], input:not([type])')];
    name(inputs[inputs.indexOf(pw) - 1], 'Email address');
    name(pw, 'Access code');
  }).observe(document.documentElement, {childList: true, subtree: true, characterData: true});
  // Gradio shows nothing when the login is refused for too many attempts (429): say so above the button.
  const fetch0 = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await fetch0(...args);
    if (response.status === 429 && String((args[0] && args[0].url) || args[0]).includes('login')) {
      const detail = (await response.clone().json().catch(() => ({}))).detail || 'Too many login attempts.';
      let note = document.getElementById('pk-login-note');
      const button = [...document.querySelectorAll('button')].find((x) => /log ?in/i.test(x.textContent));
      if (!note && button) {
        note = Object.assign(document.createElement('p'), {id: 'pk-login-note'});
        note.style.cssText = 'margin:10px 0;padding:10px 12px;border-radius:8px;color:#b91c1c;background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.4)';
        button.before(note);
      }
      if (note) note.textContent = detail;
    }
    return response;
  };
})();
</script>"""

# Shown above the login form. Inline styles: the login page is rendered outside the app, so the
# app's CSS does not reach it.
SIGNUP_CALL_TO_ACTION = (
    "<div style='text-align:center'>"
    "<p style='margin:0 0 4px'><span style='font-weight:600'>Pandemonium research preview</span>: English-only multi-speaker transcription "
    f"of recordings up to {LIMIT}.</p>"
    "<p style='margin:0 0 14px'>No access code yet? See it on a real meeting first, or request one: it is "
    "emailed to you within a minute.</p>"
    "<a href='example/' style='display:inline-block;margin:0 6px 6px;padding:11px 26px;border-radius:8px;"
    "border:1px solid #0e9f6e;color:#0e9f6e;font-weight:600;font-size:16px;text-decoration:none'>See an example</a>"
    "<a href='signup' style='display:inline-block;margin:0 6px 6px;padding:11px 26px;border-radius:8px;background:#0e9f6e;"
    "color:#fff;font-weight:600;font-size:16px;text-decoration:none'>Request an access code</a>"
    "<p style='margin:16px 0 0;font-size:13px;opacity:.8'>Already have one? Log in below with your email "
    "address and access code. "
    "<a href='privacy' style='color:inherit'>Privacy notice</a></p></div>"
)

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, RedirectResponse, Response

    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    root = (os.environ.get("GRADIO_ROOT_PATH") or "").rstrip("/")
    access.init_db()
    access.PUBLIC_URL = os.environ.get("DEMO_PUBLIC_URL", "")
    started = time.time()
    pipeline.warm_up()  # so the first visitor after a (re)start does not wait for CUDA to initialize
    print(f"Warm-up: {time.time() - started:.1f} s", flush=True)

    from fastapi.responses import FileResponse

    server = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # no public route listing
    # Gradio's own CORS layer allows any Origin unless the host is localhost (behind nginx it never is),
    # and it steps aside when the parent app configures CORS. So do: the public site's origin only,
    # never another site and never "null".
    from urllib.parse import urlsplit
    from starlette.middleware.cors import CORSMiddleware

    public = urlsplit(access.PUBLIC_URL)
    server.add_middleware(CORSMiddleware, allow_origins=[f"{public.scheme}://{public.netloc}"] if public.netloc else [],
                          allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])

    async def example_media(name: str):
        if name not in EXAMPLE_MEDIA:
            return Response(status_code=404)
        return FileResponse(EXAMPLE_MEDIA[name], filename=None if name == "audio.mp3" else name)

    server.add_api_route("/example/media/{name}", example_media, methods=["GET"])

    async def preview_image():
        return FileResponse(PREVIEW_IMAGE, media_type="image/png")

    server.add_api_route("/preview.png", preview_image, methods=["GET"])  # the link-preview card
    _add_signup_routes(server)

    @server.middleware("http")
    async def land_on_example(request: Request, call_next):
        """Visitors who are not logged in start at the example; `?login` (any query) shows the login form.

        Only a session the server still knows counts: a deleted or revoked account's cookie, or one from
        before a restart, lands on the example like any visitor's, not in an app that would refuse it.
        """
        path = request.url.path
        if path == "/login" and request.method == "POST" and _login_blocked(request):
            return JSONResponse({"detail": "Too many login attempts. Please try again in a few minutes."}, status_code=429)
        user = _session_user(request)
        if user and not _still_active(user):  # revoked (or deleted) since logging in: end the session now
            for token in [t for t, u in app_main.tokens.items() if u == user]:
                app_main.tokens.pop(token, None)
            user = None
        if request.method == "GET" and path == "/" and not request.url.query and not user:
            return RedirectResponse("example/", status_code=302)
        if _hidden(path):
            return Response(status_code=404)
        if path.startswith("/gradio_api/file/"):  # an alias of file= the pages never use
            return Response(status_code=404)
        if path.startswith("/gradio_api/file=") and not _file_allowed(path[len("/gradio_api/file="):], user):
            return Response(status_code=403)
        response = await call_next(request)
        if path == "/login" and request.method == "POST" and response.status_code != 200:
            _login_failed(request)
        response.headers.update(SECURITY_HEADERS)
        if path == "/gradio_api/upload" and request.method == "POST" and response.status_code == 200:
            body = b"".join([chunk async for chunk in response.body_iterator])
            for uploaded in json.loads(body):
                _own(uploaded, user)
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            return Response(body, status_code=200, headers=headers)  # (SECURITY_HEADERS already in `headers`)
        if path == "/login" and request.method == "POST":
            # Gradio sets the session cookie SameSite=None, i.e. sent along with requests from other sites
            # too. The app is only ever used from its own pages, so Lax.
            response.raw_headers[:] = [
                (k, re.sub(rb"(?i)samesite=none", b"SameSite=lax", v) if k.lower() == b"set-cookie" else v)
                for k, v in response.raw_headers]
        if path == "/example/config" and response.status_code == 200:
            # The public page works without knowing which Gradio this is.
            body = json.loads(b"".join([chunk async for chunk in response.body_iterator]))
            body.pop("version", None)
            return JSONResponse(body)
        if request.method == "GET" and path in ("/", "/example/") and "text/html" in response.headers.get("content-type", ""):
            body = b"".join([chunk async for chunk in response.body_iterator])
            body = re.sub(rb'"version":"[^"]*",', b"", body, count=1)  # the page's config stub names the Gradio version
            # The link preview (chat apps, social sites): ours, instead of Gradio's "Gradio · Click to try out the app!".
            body = re.sub(rb'<meta\s+(?:property|name)="(?:og:[^"]*|twitter:[^"]*|description)"[^>]*>\s*', b"", body)
            body = body.replace(b"</head>", _preview_meta().encode() + b"</head>", 1)
            if path == "/":  # the page that may show the login form: give its fields our names (LOGIN_LABELS_JS)
                body = body.replace(b"</head>", LOGIN_LABELS_JS.encode() + b"</head>", 1)
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            return Response(body, status_code=response.status_code, headers=headers)
        return response

    common = dict(
        show_error=False,  # no server tracebacks in the browser
        footer_links=[],  # no "Use via API" / built-with links
        theme=gr.themes.Ocean(),
        css=CSS + TIMELINE_CSS,
    )
    # Public and mounted first, so the app at "/" does not swallow it. It serves only the files the page uses:
    # the rest of the example folder (meta.json, README.md) says how the example was produced.
    gr.mount_gradio_app(server, explore.queue(default_concurrency_limit=8), path="/example",
                        root_path=f"{root}/example" if root else None,
                        max_file_size="1kb",  # it takes no uploads (and `_hidden` refuses the route anyway)
                        **common)
    if root:
        mount = next(r for r in server.routes if getattr(r, "path", None) == "/example")
        mount.app = _WithPrefix(mount.app, root)
    gr.mount_gradio_app(
        server, demo.queue(default_concurrency_limit=1), path="/", root_path=root or None,
        # Gates the page AND every API route, always: each person logs in with their own code.
        auth=_check,
        auth_message=SIGNUP_CALL_TO_ACTION,
        js=SESSION_EXPIRED_JS,
        max_file_size="1gb",  # 40 min of 48 kHz stereo 24-bit WAV is ~0.7 GB
        allowed_paths=[RESULTS_DIR],  # the exports; each served only to its owner (`_file_allowed`)
        **common,
    )
    app_main = next(r.app for r in server.routes if getattr(r, "path", None) == "")  # the logged-in app
    _sessions = app_main.tokens
    threading.Thread(target=_clean_uploads, daemon=True).start()

    def _session_user(request):
        """The account a request's Gradio session cookie belongs to, if any."""
        token = (request.cookies.get(f"access-token-{app_main.cookie_id}")
                 or request.cookies.get(f"access-token-unsecure-{app_main.cookie_id}"))
        return app_main.tokens.get(token) if token else None

    print(f"Example page: {(access.PUBLIC_URL or f'http://{host}:{port}/').rstrip('/')}/example/", flush=True)
    print(f"Signup page: {(access.PUBLIC_URL or f'http://{host}:{port}/').rstrip('/')}/signup", flush=True)
    # Open pages hold a heartbeat stream that never ends, so without a timeout a stop/restart waits on
    # them until systemd kills the process (90 s) and marks the restart failed.
    uvicorn.run(server, host=host, port=port, timeout_graceful_shutdown=3)
