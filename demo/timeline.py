"""Scrollable, zoomable diarization + transcript timeline with its own synchronized player.

One lane per speaker: the diarizer's soft probability as a faint curve, and on top of it one block
per turn carrying its transcript (or, before transcription, the diarizer's segments). The timeline
owns the only audio player on the page, streaming the upload itself, so there is one playback
position and everything follows it: a playhead moves across the lanes, the turns being spoken light
up and are shown in a "now playing" bar, clicking or dragging anywhere seeks and plays, and clicking
a block plays just that turn.

Positions are in seconds scaled by CSS variables (`--pps` pixels per second, `--t` the playhead),
so zooming and playback are style changes on the client: nothing re-renders, nothing goes to the
server. Other components play a span through `window.pkTimeline.play(start, end)`.
"""

import html
from urllib.parse import quote

import numpy as np

LABEL_WIDTH = 128  # px, the sticky speaker-name column (room for the name and its ✎)
HELP = (
    "<div class='tl-help' hidden><div class='tl-help-card'><b>Keyboard and mouse</b><dl>"
    "<dt>Space</dt><dd>play / pause</dd>"
    "<dt>← / →</dt><dd>back / forward 5 s</dd>"
    "<dt>Click a block</dt><dd>play that turn</dd>"
    "<dt>Click or drag in a lane</dt><dd>play from there</dd>"
    "<dt>Ctrl / ⌘ + wheel</dt><dd>zoom in and out</dd>"
    "<dt>Click a speaker's name</dt><dd>collapse / expand their lane</dd>"
    "<dt>?</dt><dd>show / hide this help</dd>"
    "</dl><button data-act='help'>Close</button></div></div>"
)
MAX_CURVE_POINTS = 6000  # per lane; long recordings are max-pooled down to this
NOW_ROWS_FIXED = 3  # lines the "now playing" bar always reserves at most; a 4th simultaneous turn grows it


def _pool_max(x: np.ndarray, n_out: int) -> np.ndarray:
    """Max-pool the last axis to at most `n_out` bins."""
    if x.size == 0:  # e.g. no speakers at all
        return np.zeros(x.shape[:-1] + (0,), x.dtype)
    factor = max(1, int(np.ceil(x.shape[-1] / n_out)))
    pad = (-x.shape[-1]) % factor
    if pad:
        x = np.concatenate([x, np.zeros(x.shape[:-1] + (pad,), x.dtype)], axis=-1)
    return x.reshape(*x.shape[:-1], -1, factor).max(-1)


def _area_path(values: np.ndarray, duration: float, mirrored: bool = False) -> str:
    """SVG path in a `0..duration x 0..1` viewBox: a filled area under `values` (0..1)."""
    t = (np.arange(len(values)) + 0.5) * duration / len(values)
    if mirrored:  # waveform envelope, symmetric around the middle
        top = " ".join(f"{a:.3f},{0.5 - 0.48 * v:.3f}" for a, v in zip(t, values))
        bottom = " ".join(f"{a:.3f},{0.5 + 0.48 * v:.3f}" for a, v in zip(t[::-1], values[::-1]))
        return f"M0,0.5 L{top} L{duration:.3f},0.5 L{bottom} Z"
    line = " ".join(f"{a:.3f},{1 - 0.92 * v:.3f}" for a, v in zip(t, values))
    return f"M0,1 L{line} L{duration:.3f},1 Z"


def _fmt(t: float) -> str:
    m, s = divmod(round(t, 1), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def audio_url(path: str) -> str:
    """URL the page streams an uploaded file from (Gradio's file route; relative, so it works under
    a root path and through the share relay, and it stays behind the app's auth)."""
    return "gradio_api/file=" + quote(path)


def render_timeline(audio, sample_rate, diar, names, colors, utts=None, src=None, open_at=0.0,
                    reference=None, renamable=False, hidden=()) -> str:
    """Timeline HTML.

    Args:
        audio: the waveform (for the envelope lane and the duration).
        diar: `pipeline.Diarization`.
        names / colors: per speaker row.
        utts: transcribed turns (`pipeline.utterances`); `None` shows the diarizer's segments.
        src: URL of the uploaded audio, for the built-in player.
        open_at: where the playhead and the view open, in seconds (e.g. where an example's meeting begins).
        reference: human reference turns in `utts` form, speakers mapped onto the rows: adds a hidden
            reference lane under each speaker and a toolbar toggle that shows them.
        renamable: put a rename button (✎) on each speaker's label (see `window.pkRename`).
        hidden: speaker rows not to draw (merged into another speaker).
    """
    duration = len(audio) / sample_rate
    env = _pool_max(np.abs(audio), MAX_CURVE_POINTS)
    env = env / (env.max() + 1e-9)
    probs = _pool_max(diar.probs, MAX_CURVE_POINTS)

    if utts is not None:
        blocks = [(u["start"], u["end"], u["speaker"], u["text"], u.get("word_starts")) for u in utts]
    else:
        blocks = [(s, e, spk, "", None) for s, e, spk in diar.segments]
    # The "now playing" bar keeps one line per simultaneous turn, so the page does not jump as speakers
    # come and go: as many as this recording ever has at once, up to 3 (beyond that it grows).
    events = sorted([(b[0], 1) for b in blocks] + [(b[1], -1) for b in blocks])
    live = most = 0
    for _, step in events:
        live += step
        most = max(most, live)
    now_rows = max(1, min(most, NOW_ROWS_FIXED))

    def svg(path, cls):
        return (f"<svg class='{cls}' viewBox='0 0 {duration:.3f} 1' preserveAspectRatio='none'>"
                f"<path d='{path}'/></svg>")

    lanes = [
        f"<div class='tl-lane tl-wave'><div class='tl-label'>audio</div>"
        f"{svg(_area_path(env, duration, mirrored=True), 'tl-curve')}</div>"
    ]
    def seg(start, end, spk, text, word_starts=None, ref=False):
        tip = html.escape(("Human reference\n" if ref else "") + f"{_fmt(start)} – {_fmt(end)}"
                          + (f"\n{text}" if text else ""), quote=True)
        # Width is set by the client (`fitBlocks`): a short turn widens to fit its text, into the
        # silence after it; `.tl-true` underlines the span the turn really has.
        return (f"<div class='tl-seg{' tl-refseg' if ref else ''}' data-s='{start:.2f}' data-e='{end:.2f}' "
                f"data-k='{spk}' title='{tip}' "
                # each word's start, relative to the turn: the now-playing bar follows along word by word
                + (f"data-wt='{','.join(f'{t - start:.2f}' for t in word_starts)}' " if word_starts else "") +
                f"style='left:calc(var(--lw) + var(--pps) * {start:.3f}px);--d:{max(end - start, 0.02):.3f};"
                f"width:calc(var(--pps) * {max(end - start, 0.05):.3f}px)'>"
                f"<span>{html.escape(text)}</span><i class='tl-true'></i></div>")

    for spk, name in enumerate(names):
        if spk in hidden:
            continue
        segs = [seg(start, end, s, text, ws) for start, end, s, text, ws in blocks if s == spk]
        rename = (f"<button class='tl-rename' data-rename='{spk}' title='Rename this speaker'>✎</button>"
                  if renamable else "")
        lanes.append(
            f"<div class='tl-lane' style='--c:{colors[spk]}' data-k='{spk}' data-name='{html.escape(name, quote=True)}'>"
            f"<div class='tl-label' title='Click to collapse / expand'><span class='tl-caret'>▾</span>"
            f"<span class='tl-name' data-spk-k='{spk}'>{html.escape(name)}</span>{rename}</div>"
            f"{svg(_area_path(probs[spk], duration), 'tl-curve tl-prob')}{''.join(segs)}</div>"
        )
        if reference is not None:  # right under the speaker's own lane, shown by the toolbar toggle
            refs = [seg(u["start"], u["end"], spk, u["text"], ref=True) for u in reference if u["speaker"] == spk]
            lanes.append(
                f"<div class='tl-lane tl-ref' style='--c:{colors[spk]}' data-k='{spk}' "
                f"data-name='{html.escape(name + ' · reference', quote=True)}'>"
                f"<div class='tl-label' title='Human reference for {html.escape(name, quote=True)}'>reference</div>"
                f"{''.join(refs)}</div>"
            )

    player = (
        f"<audio class='tl-audio' preload='metadata' src='{html.escape(src, quote=True)}'></audio>"
        if src else ""
    )
    hint = ("Click in a lane to play from there · press ? for shortcuts" if utts is None
            else "Click a block to play that turn · press ? for shortcuts")
    return (
        f"<div class='tl{' tl-diar-only' if utts is None else ''}' data-dur='{duration:.3f}' data-start='{open_at:.2f}' style='--lw:{LABEL_WIDTH}px;--dur-s:{duration:.3f};--now-rows:{now_rows}'>"
        f"{player}"
        "<div class='tl-bar'>"
        "<button class='tl-play' data-act='play' title='Play / pause (Space)'>▶</button>"
        f"<span class='tl-time'>0:00.0 / {_fmt(duration)}</span>"
        "<span class='tl-gap'></span>"
        "<button data-zoom='out' title='Zoom out'>−</button>"
        "<button data-zoom='in' title='Zoom in'>+</button><button data-zoom='fit'>Fit</button>"
        "<select class='tl-rate' title='Playback speed'><option value='1'>1×</option><option value='1.25'>1.25×</option>"
        "<option value='1.5'>1.5×</option><option value='2'>2×</option></select>"
        "<label class='tl-follow' title='Keep the playhead in view'><input type='checkbox' checked> Follow</label>"
        + ("<label class='tl-follow tl-reftoggle' title='Show the human transcript under each speaker'>"
           "<input type='checkbox'> Human reference</label>" if reference is not None else "") +
        f"<span class='tl-hint'>{hint}</span>"
        "<button class='tl-helpbtn' data-act='help' title='Keyboard shortcuts (?)'>?</button></div>"
        f"{HELP}"
        "<div class='tl-now'><span class='tl-now-idle'>Press ▶ or click the timeline to listen.</span></div>"
        "<div class='tl-scroll' tabindex='0'><div class='tl-canvas'>"
        "<div class='tl-ruler'><div class='tl-label'></div><div class='tl-ticks'></div></div>"
        f"{''.join(lanes)}<div class='tl-head'></div></div></div></div>"
    )


CSS = """
.tl { --pps: 20; --t: 0; font-size: 13px; }
.tl-bar { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.tl-bar button { min-width: 32px; height: 30px; padding: 0 10px; border-radius: 6px; cursor: pointer;
                 border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill);
                 color: var(--body-text-color); font-size: 14px; }
.tl-bar button:hover { background: var(--button-secondary-background-fill-hover); }
.tl-bar .tl-play { min-width: 44px; font-size: 15px; background: var(--button-primary-background-fill);
                   color: var(--button-primary-text-color); border-color: transparent; }
.tl-time { font-family: var(--font-mono); font-variant-numeric: tabular-nums; min-width: 9.5em;
           color: var(--body-text-color); }
.tl-gap { width: 8px; }
.tl-follow { display: inline-flex; align-items: center; gap: 4px; margin-left: 6px; cursor: pointer;
             color: var(--body-text-color-subdued); user-select: none; }
.tl-hint { margin-left: 8px; color: var(--body-text-color-subdued); }
.tl-now { min-height: calc(14px + var(--now-rows, 1) * 22px); margin-bottom: 6px; padding: 6px 10px; border-radius: 8px;
          box-sizing: border-box; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
          display: flex; flex-direction: column; font-size: 14px; line-height: 22px; }
.tl-now-idle { color: var(--body-text-color-subdued); font-size: 13px; }
.tl-now-row { display: flex; gap: 10px; align-items: baseline; min-width: 0; }
.tl-now-row b { color: var(--c); font-weight: 600; white-space: nowrap; font-family: inherit; }
.tl-now-row span { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tl-scroll { overflow-x: auto; overflow-y: hidden; border: 1px solid var(--border-color-primary);
             border-radius: 8px; background: var(--background-fill-primary); outline: none;
             cursor: text; user-select: none; touch-action: pan-x; }
.tl-canvas { position: relative; width: calc(var(--lw) + var(--pps) * var(--dur-s) * 1px); min-width: 100%; }
.tl-lane, .tl-ruler { position: relative; height: 62px; border-top: 1px solid var(--border-color-primary); }
.tl-ruler { height: 24px; border-top: none; cursor: ew-resize; }
.tl-wave { height: 40px; }
.tl-label { position: sticky; left: 0; z-index: 5; width: var(--lw); height: 100%; box-sizing: border-box;
            display: flex; align-items: center; padding: 0 10px; font-weight: 600; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; color: var(--c, var(--body-text-color-subdued));
            background: var(--background-fill-primary); border-right: 1px solid var(--border-color-primary);
            cursor: default; }
.tl-lane[data-name]:not(.tl-ref) > .tl-label { cursor: pointer; }
/* Human reference lanes: hidden until the toolbar toggle shows them, dashed to tell them apart. */
.tl:not(.show-ref) .tl-ref { display: none; }
.tl-ref { height: 48px; border-top-style: dashed; }
.tl-ref > .tl-label { cursor: default; font-weight: 500; font-style: italic; font-size: 12px; padding-left: 25px; }
/* Longhands: a `border` shorthand holding var() followed by `border-left` was dropped by the browser. */
.tl-ref .tl-seg { top: 5px; bottom: 7px; background: var(--background-fill-primary);
                  border-width: 1.5px 1.5px 1.5px 3px; border-style: dashed dashed dashed solid; border-color: var(--c); }
.tl-ref .tl-seg .tl-true { display: none; }
.tl-ref .tl-seg:hover { background: color-mix(in srgb, var(--c) 10%, var(--background-fill-primary)); }
.tl-reftoggle { color: var(--body-text-color); }
.tl-rate { height: 30px; padding: 0 6px; border-radius: 6px; border: 1px solid var(--border-color-primary);
           background: var(--button-secondary-background-fill); color: var(--body-text-color); font-size: 13px; }
.tl-bar .tl-helpbtn { min-width: 30px; padding: 0; border-radius: 50%; font-weight: 600; }
.tl-help { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center;
           padding: 16px; background: rgba(0, 0, 0, .4); }
.tl-help[hidden] { display: none; }
.tl-help-card { max-width: 420px; width: 100%; padding: 20px 24px; border-radius: 12px; font-size: 14px;
                background: var(--background-fill-primary); color: var(--body-text-color);
                border: 1px solid var(--border-color-primary); box-shadow: 0 10px 30px rgba(0, 0, 0, .25); }
.tl-help-card b { font-size: 16px; font-weight: 600; }
.tl-help-card dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; margin: 12px 0 16px; }
.tl-help-card dt { font-family: var(--font-mono); font-size: 13px; white-space: nowrap; }
.tl-help-card dd { margin: 0; color: var(--body-text-color-subdued); }
.tl-help-card button { padding: 6px 16px; border-radius: 6px; cursor: pointer; border: 1px solid var(--border-color-primary);
                       background: var(--button-secondary-background-fill); color: var(--body-text-color); }
.tl-name { overflow: hidden; text-overflow: ellipsis; }
.tl-rename { flex: none; margin-left: 4px; padding: 1px 3px; border: 0; border-radius: 4px; background: none;
             cursor: pointer; font-size: 12px; color: var(--body-text-color-subdued); opacity: .6; }
.tl-rename:hover, .tl-rename:focus { opacity: 1; background: var(--background-fill-secondary); }
.tl-now-row mark { background: color-mix(in srgb, var(--c) 28%, transparent); color: inherit; border-radius: 3px;
                   padding: 0 2px; }
/* Search in the Conversation tab: matching turns stand out, the rest fade. */
.tl.searching .tl-seg:not(.tl-match):not(.tl-refseg) { opacity: .3; }
.tl-seg.tl-match { outline: 2px solid var(--color-accent, #f97316); outline-offset: 1px; z-index: 2; }
/* With the reference shown, each speaker and their reference read as one group, with air between groups. */
.tl.show-ref .tl-lane[data-name]:not(.tl-ref) { margin-top: 12px; border-top: 1px solid var(--border-color-primary); }
.tl.show-ref .tl-ref { height: 52px; border-top: 1px dashed color-mix(in srgb, var(--c) 35%, transparent);
                       border-bottom: 1px solid var(--border-color-primary);
                       background: color-mix(in srgb, var(--c) 5%, var(--background-fill-primary)); }
.tl.show-ref .tl-ref > .tl-label { background: color-mix(in srgb, var(--c) 5%, var(--background-fill-primary)); }
.tl.show-ref .tl-ref .tl-seg { top: 8px; bottom: 9px; }
.tl-caret { display: inline-block; width: 12px; margin-right: 3px; font-size: 10px; transition: transform .15s; }
.tl-lane.collapsed { height: 22px; }
.tl-lane.collapsed .tl-caret { transform: rotate(-90deg); }
.tl-lane.collapsed .tl-label { font-size: 12px; }
.tl-lane.collapsed .tl-seg { top: 2px; bottom: 9px; padding: 0; border-left-width: 2px; }
.tl-lane.collapsed .tl-seg span { display: none; }
.tl-lane.collapsed .tl-prob { height: 7px; bottom: 1px; }
.tl-ticks { position: absolute; left: var(--lw); top: 0; bottom: 0; right: 0; }
.tl-ticks span { position: absolute; top: 4px; transform: translateX(-50%); white-space: nowrap;
                 font-family: var(--font-mono); font-size: 11px; color: var(--body-text-color-subdued); }
.tl-ticks span:first-child { transform: none; }
.tl-ticks span::after { content: ""; position: absolute; left: 50%; top: 16px; height: 4px;
                        border-left: 1px solid var(--border-color-primary); }
.tl-curve { position: absolute; left: var(--lw); top: 0; height: 100%;
            width: calc(var(--pps) * var(--dur-s) * 1px); pointer-events: none; }
.tl-curve path { fill: var(--body-text-color-subdued); opacity: 0.35; }
/* Speakers found, nothing transcribed yet: the probability is the content, over the whole lane; the
   diarizer's segments stay (hidden) so the now-playing bar can still say who is speaking. */
.tl-diar-only .tl-lane .tl-prob { top: 6px; bottom: 4px; height: auto; }
.tl-diar-only .tl-lane .tl-prob path { opacity: .55; }
.tl-diar-only .tl-seg { visibility: hidden; }
/* The diarizer's speech probability: its own strip along the bottom of the lane, under the turns. */
.tl-prob { top: auto; bottom: 2px; height: 14px; }
.tl-prob path { fill: var(--c); opacity: 0.45; }
.tl-seg { position: absolute; top: 5px; bottom: 19px; min-width: 4px; box-sizing: border-box; cursor: pointer;
          border-radius: 4px; border-left: 3px solid var(--c); overflow: hidden;
          background: color-mix(in srgb, var(--c) 18%, var(--background-fill-primary));
          color: var(--body-text-color); line-height: 1.3; padding: 2px 5px; font-size: 12px; }
.tl-seg span { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
               overflow-wrap: normal; word-break: normal; }
/* Narrower than its first word: one line, cut with an ellipsis, rather than a word broken in two. */
.tl-seg.clip span { display: block; white-space: nowrap; text-overflow: ellipsis; }
.tl-seg.tiny span { display: none; }  /* no room for even one word: hover or zoom in instead */
.tl-true { position: absolute; left: 0; bottom: 0; height: 3px; width: calc(var(--pps) * var(--d) * 1px);
           background: var(--c); opacity: 0.85; }
.tl-seg:hover { background: color-mix(in srgb, var(--c) 32%, var(--background-fill-primary)); z-index: 2; }
.tl-seg.live { background: color-mix(in srgb, var(--c) 42%, var(--background-fill-primary)); z-index: 2; }
.tl-seg.on { outline: 2px solid var(--c); outline-offset: 1px; z-index: 2; }
.tl-head { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; z-index: 4; pointer-events: none;
           left: var(--lw); background: var(--color-accent, #f97316); will-change: transform; }
.tl-head::before { content: ""; position: absolute; top: 0; left: -5px; border: 6px solid transparent;
                   border-top: 9px solid var(--color-accent, #f97316); }
@media (max-width: 640px) {
  .tl { --lw: 82px !important; }
  .tl-label { padding: 0 6px; font-size: 12px; }
  .tl-lane > .tl-label .tl-caret { display: none; }  /* every pixel of the name column counts on a phone */
  .tl-rename { display: none; }  /* no room (and no hover) on a phone: rename in the Per speaker tab */
  .tl-hint { display: none; }
}
"""

# Runs once per page load; the template is re-rendered on every new value, so events are bound by
# delegation on `element` and the per-recording state is rebuilt through `watch('value', ...)`.
JS = r"""
const MAX_PPS = 300;
// Played around a selected turn, so its first and last words are not clipped (word times are 80 ms frames).
const LEAD = 0.1, TAIL = 0.15;
const SKIP = 5;  // s per ← / → press
const tl = () => element.querySelector('.tl');
const audio = () => element.querySelector('.tl-audio');
// Read once per layout (the label column narrows on a phone): getComputedStyle in the playback loop would
// force a style pass every frame.
const lw = (t) => t._lw ?? (t._lw = parseFloat(getComputedStyle(t).getPropertyValue('--lw')) || 0);
window.addEventListener('resize', () => { const t = tl(); if (t) t._lw = undefined; });
// The playhead moves by transform only. (It used to follow a `--t` variable on the timeline, which every
// block inherits: each frame then re-styled the whole timeline, ~35 ms, and playback stuttered.)
const placeHead = (t) => {
  const head = t.querySelector('.tl-head');
  if (head) head.style.transform = `translateX(${(element._x || 0) * element._pps}px)`;
};
const fit = (t) => Math.max(0.05, (t.querySelector('.tl-scroll').clientWidth - lw(t) - 4) / parseFloat(t.dataset.dur));
const fmt = (x, tenths) => {
  const h = Math.floor(x / 3600), m = Math.floor((x % 3600) / 60), s = x % 60;
  const ss = tenths ? s.toFixed(1).padStart(4, '0') : String(Math.floor(s)).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
};
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

const ruler = (t) => {
  const pps = element._pps, dur = parseFloat(t.dataset.dur);
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => s * pps >= 70) || 3600;
  let h = '';
  for (let x = 0; x <= dur; x += step) h += `<span style="left:${x * pps}px">${fmt(x)}</span>`;
  t.querySelector('.tl-ticks').innerHTML = h;
};
const applyZoom = (t) => {
  const pps = element._pps;
  t.style.setProperty('--pps', pps);
  t.style.setProperty('--dur-s', t.dataset.dur);
  ruler(t);
  fitBlocks(pps);
  placeHead(t);
};
// Each block is its turn's span, widened when that is too narrow for its text: into the silence before
// the speaker's next turn, never over it. Recomputed on every zoom.
let measureCtx = null;
const textWidth = (el, text) => {
  if (!measureCtx) {
    measureCtx = document.createElement('canvas').getContext('2d');
    const cs = getComputedStyle(el);
    measureCtx.font = `${cs.fontSize} ${cs.fontFamily}`;
  }
  return measureCtx.measureText(text).width;
};
const fitBlocks = (pps) => {
  const lanes = new Map();
  (element._blocks || []).forEach((b) => { if (!lanes.has(b.lane)) lanes.set(b.lane, []); lanes.get(b.lane).push(b); });
  lanes.forEach((blocks) => {
    blocks.sort((p, q) => p.s - q.s);
    blocks.forEach((b, j) => {
      const real = (b.e - b.s) * pps;
      const room = j + 1 < blocks.length ? (blocks[j + 1].s - b.s) * pps - 3 : Infinity;
      const want = b.text ? Math.min(b.tw + 14, 160) : 0;  // 14 = border + padding
      const w = Math.max(real, 4, Math.min(want, room));
      b.el.style.width = `${w}px`;
      b.el.classList.toggle('tiny', w < 18);
      b.el.classList.toggle('clip', w >= 18 && w < b.fw + 14);
    });
  });
};
const zoom = (t, how, anchor) => {
  const sc = t.querySelector('.tl-scroll'), L = lw(t), old = element._pps, f = fit(t);
  const next = how === 'fit' ? f : Math.min(MAX_PPS, Math.max(Math.min(f, MAX_PPS), old * how));
  const ax = anchor ?? sc.clientWidth / 2;
  const sec = (sc.scrollLeft + ax - L) / old;  // keep the time under the anchor in place
  element._pps = next;
  applyZoom(t);
  sc.scrollLeft = Math.max(0, sec * next + L - ax);
};

// --- playback -------------------------------------------------------------------------------
const setTime = (t, x) => {
  element._x = x;
  placeHead(t);
  const a = audio();
  const time = `${fmt(x, true)} / ${fmt(parseFloat(t.dataset.dur), true)}`, clock = t.querySelector('.tl-time');
  if (clock.textContent !== time) clock.textContent = time;
  // Turns being spoken: light them up and show them, only touching the DOM when the set changes.
  const spoken = element._blocks.filter((b) => b.s <= x && x < b.e);
  const live = spoken.filter((b) => !b.ref);  // the now-playing bar shows the model's turns only
  // One line per turn: the part of its text around where the playhead is in it (a turn's words are
  // spread evenly over its span, close enough to follow along), cut at a word boundary.
  // With word times (the model's turns): the word being spoken, highlighted, with a few before it.
  // Without (reference turns): the text near where the playhead is, assuming evenly spread words.
  const wordAt = (b) => {
    if (!b.wt) return -1;
    let i = 0;
    while (i + 1 < b.wt.length && b.s + b.wt[i + 1] <= x) i++;
    return i;
  };
  const from = (b) => {
    if (!b.text) return 0;
    if (b.wt) return wordAt(b);
    const at = Math.floor((x - b.s) / Math.max(b.e - b.s, 1e-3) * b.text.length) - 30;
    return at <= 0 ? 0 : b.text.lastIndexOf(' ', at) + 1;
  };
  const line = (b, k) => {
    if (!b.wt) return `${k ? '… ' : ''}${esc(b.text.slice(k))}`;
    const words = b.text.split(' '), first = Math.max(0, k - 4);
    return (first ? '… ' : '') + words.slice(first).map((w, j) => first + j === k ? `<mark>${esc(w)}</mark>` : esc(w)).join(' ');
  };
  const starts = live.map(from);
  const playing = !!(a && !a.paused);
  // Which turns are spoken changes a few times a minute; the word in them a few times a second. Only the
  // former touches the blocks and the transcript.
  const turns = spoken.map((b) => b.i).join(',') + (playing ? '+' : '');
  if (turns !== element._turnsKey) {
    element._turnsKey = turns;
    (element._lit || []).forEach((b) => { if (!spoken.includes(b)) b.el.classList.remove('live'); });
    spoken.forEach((b) => b.el.classList.add('live'));
    element._lit = spoken;
    highlightTranscript(spoken, playing);
  }
  const key = live.map((b, j) => `${b.i}:${starts[j]}`).join(',') + (playing ? '+' : '');
  if (key !== element._liveKey) {
    element._liveKey = key;
    const now = t.querySelector('.tl-now');
    const rows = live.map((b, j) => [b, starts[j]]).filter(([b]) => b.text).map(([b, k]) =>
      `<div class="tl-now-row" style="--c:${b.color}"><b>${esc(b.name)}</b><span>${line(b, k)}</span></div>`);
    if (rows.length) now.innerHTML = rows.join('');
    else if (live.length) now.innerHTML = live.map((b) =>
      `<div class="tl-now-row" style="--c:${b.color}"><b>${esc(b.name)}</b><span class="tl-now-idle">speaking (transcribe to see the words)</span></div>`).join('');
    else now.innerHTML = `<span class="tl-now-idle">${a && !a.paused ? '(no one speaking)' : 'Press ▶ or click the timeline to listen.'}</span>`;
  }
  // Follow: keep the playhead on screen while playing, turning the page when it reaches the edge.
  const follow = t.querySelector('.tl-follow input');
  if (a && !a.paused && follow && follow.checked && !element._dragging) {
    const sc = t.querySelector('.tl-scroll'), px = lw(t) + x * element._pps;
    if (px < sc.scrollLeft + lw(t) || px > sc.scrollLeft + sc.clientWidth - 40) sc.scrollLeft = px - lw(t) - 40;
  }
};
// The same turns in the Conversation / Per speaker views (rows carry the turn's times, as the blocks do).
const highlightTranscript = (live, playing) => {
  document.querySelectorAll('.pk-turn.pk-live').forEach((row) => row.classList.remove('pk-live'));
  const follow = tl() && tl().querySelector('.tl-follow input');
  live.forEach((b) => {
    document.querySelectorAll(`.pk-turn[data-s="${b.el.dataset.s}"][data-e="${b.el.dataset.e}"]`).forEach((row) => {
      row.classList.add('pk-live');
      const box = row.closest('.pk-transcript');
      // A hidden tab or a collapsed speaker: highlight it, but do not scroll to it.
      if (!playing || !(follow && follow.checked) || !box || !box.offsetParent || !row.offsetParent) return;
      const r = row.getBoundingClientRect(), v = box.getBoundingClientRect();
      if (r.top < v.top || r.bottom > v.bottom) box.scrollTop += r.top - v.top - v.height / 3;
    });
  });
};
const tick = () => {
  const t = tl(), a = audio();
  if (!t || !a) return;
  if (element._stopAt != null && a.currentTime >= element._stopAt) { a.pause(); element._stopAt = null; }
  setTime(t, a.currentTime);
  if (!a.paused) element._raf = requestAnimationFrame(tick);
};
const playIcon = () => {
  const t = tl(), a = audio();
  if (t && a) t.querySelector('.tl-play').textContent = a.paused ? '▶' : '❚❚';
};
const seek = (x, play, stopAt) => {
  const t = tl(), a = audio();
  if (!t || !a) return;
  x = Math.max(0, Math.min(parseFloat(t.dataset.dur), x));
  a.currentTime = x;
  element._stopAt = stopAt ?? null;
  setTime(t, x);
  if (play) a.play().catch((err) => console.warn('pk play blocked:', err.name, err.message));
};
const toggleHelp = (show) => {
  const h = element.querySelector('.tl-help');
  if (h) h.hidden = show === undefined ? !h.hidden : !show;
};
// Rename speaker k everywhere on the page (timeline, transcript views, the now-playing bar) and tell
// the server (a hidden box, #pk-names), which renames it in the table and the downloads too.
window.pkRename = (k) => {
  const shown = document.querySelector(`[data-spk-k="${k}"]`);
  const current = shown ? shown.textContent : `Speaker ${+k + 1}`;
  const typed = window.prompt(`Rename ${current} to:`, current);
  const name = typed && typed.trim().slice(0, 40);
  if (!name) return;
  document.querySelectorAll(`[data-spk-k="${k}"]`).forEach((el) => { el.textContent = name; });
  document.querySelectorAll(`.tl-lane[data-k="${k}"]:not(.tl-ref)`).forEach((lane) => { lane.dataset.name = name; });
  (element._blocks || []).forEach((b) => { if (b.lane.dataset.k === String(k) && !b.ref) b.name = name; });
  element._liveKey = null;
  const box = document.querySelector('#pk-names textarea, #pk-names input');
  if (box) {
    let names = {};
    try { names = JSON.parse(box.value || '{}'); } catch (err) { names = {}; }
    names[k] = name;
    box.value = JSON.stringify(names);
    box.dispatchEvent(new Event('input', {bubbles: true}));  // updates the box's value in Gradio's state
    const send = document.getElementById('pk-names-apply');  // Gradio ignores scripted input events; a click it sends
    if (send) setTimeout(() => send.click(), 50);
  }
};
const togglePlay = () => {
  const a = audio();
  if (!a) return;
  if (a.paused) { element._stopAt = null; a.play().catch((err) => console.warn('pk play blocked:', err.name, err.message)); }
  else a.pause();
};
const timeAt = (t, clientX) => {
  const sc = t.querySelector('.tl-scroll');
  return (clientX - sc.getBoundingClientRect().left + sc.scrollLeft - lw(t)) / element._pps;
};

const init = () => {
  const t = tl();
  if (!t) return;
  const dur = parseFloat(t.dataset.dur);
  let opened = false;
  if (!element._pps || element._dur !== dur) {  // new recording: fit if readable, else 20 px/s
    element._dur = dur;
    element._pos = parseFloat(t.dataset.start) || 0;
    const f = fit(t);
    element._pps = f >= 12 ? f : 20;
    element._collapsed = new Set();  // speaker lanes folded away; kept across re-renders of the same recording
    opened = true;
  }
  element._collapsed = element._collapsed || new Set();
  const rate = t.querySelector('.tl-rate');
  if (rate) {
    rate.value = String(element._rate || 1);
    rate.addEventListener('change', () => { element._rate = parseFloat(rate.value); const a = audio(); if (a) a.playbackRate = element._rate; });
  }
  const refToggle = t.querySelector('.tl-reftoggle input');
  if (refToggle) {
    refToggle.checked = !!element._showRef;
    t.classList.toggle('show-ref', refToggle.checked);
    refToggle.addEventListener('change', () => { element._showRef = refToggle.checked; t.classList.toggle('show-ref', refToggle.checked); });
  }
  t.querySelectorAll('.tl-lane[data-name]').forEach((lane) => lane.classList.toggle('collapsed', element._collapsed.has(lane.dataset.name)));
  element._blocks = [...t.querySelectorAll('.tl-seg')].map((el, i) => {
    const lane = el.closest('.tl-lane');
    const text = el.textContent;
    return {i, el, lane, ref: el.classList.contains('tl-refseg'),
            s: parseFloat(el.dataset.s), e: parseFloat(el.dataset.e), text, tw: text ? textWidth(el, text) : 0,
            fw: text ? textWidth(el, text.split(' ')[0]) : 0,
            wt: el.dataset.wt ? el.dataset.wt.split(',').map(Number) : null,
            name: lane.dataset.name, color: lane.style.getPropertyValue('--c')};
  });
  element._liveKey = element._turnsKey = null;
  element._lit = [];
  t._lw = undefined;
  applyZoom(t);
  const a = audio();
  if (a) {
    a.addEventListener('play', () => { playIcon(); cancelAnimationFrame(element._raf); tick(); });
    a.addEventListener('pause', () => { playIcon(); tick(); });
    a.addEventListener('seeked', tick);
    a.addEventListener('timeupdate', () => { element._pos = a.currentTime; });
    a.playbackRate = element._rate || 1;
    a.addEventListener('loadedmetadata', () => { a.playbackRate = element._rate || 1; });
    // A re-render (e.g. after Transcribe) replaces the player: carry the position over.
    if (element._pos) a.addEventListener('loadedmetadata', () => { a.currentTime = element._pos; tick(); }, {once: true});
  }
  setTime(t, element._pos || 0);
  // Open the view where the playhead starts, a little before it.
  if (opened && element._pos) t.querySelector('.tl-scroll').scrollLeft = Math.max(0, (element._pos - 2) * element._pps);
  playIcon();
};

// One public entry point, for the transcript views and the table.
window.pkTimeline = {
  play: (start, end) => {
    const t = tl();
    if (!t) return;
    element.querySelectorAll('.tl-seg.on').forEach((x) => x.classList.remove('on'));
    const b = (element._blocks || []).find((b) => Math.abs(b.s - start) < 0.01 && Math.abs(b.e - end) < 0.01);
    if (b) b.el.classList.add('on');
    // A little context either side, so even a one-word turn is audible.
    seek(Math.max(0, start - LEAD), true, end + TAIL);
    t.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  },
};

element.addEventListener('click', (e) => {
  const t = tl();
  if (!t) return;
  const z = e.target.closest('[data-zoom]');
  if (z) { zoom(t, z.dataset.zoom === 'in' ? 1.6 : z.dataset.zoom === 'out' ? 1 / 1.6 : 'fit'); return; }
  if (e.target.closest('[data-act=play]')) togglePlay();
  if (e.target.closest('[data-act=help]')) { toggleHelp(); return; }
  if (e.target.closest('.tl-help') && !e.target.closest('.tl-help-card')) { toggleHelp(false); return; }
  const rename = e.target.closest('[data-rename]');
  if (rename) { e.stopPropagation(); window.pkRename(rename.dataset.rename); return; }
  const label = e.target.closest('.tl-lane[data-name]:not(.tl-ref) > .tl-label');
  if (label) {
    const lane = label.parentElement, name = lane.dataset.name;
    lane.classList.toggle('collapsed');
    if (lane.classList.contains('collapsed')) element._collapsed.add(name); else element._collapsed.delete(name);
  }
});

// Click a block: play that turn. Click / drag anywhere else on the timeline: scrub, play on release.
element.addEventListener('pointerdown', (e) => {
  const t = tl();
  if (!t || e.button !== 0 || !e.target.closest('.tl-scroll') || e.target.closest('.tl-label')) return;
  // The scrollbar belongs to `.tl-scroll` itself (the lanes cover the rest of it): leave it to scroll.
  const sc = t.querySelector('.tl-scroll'), box = sc.getBoundingClientRect();
  if (e.target === sc || e.clientY > box.top + sc.clientTop + sc.clientHeight) return;
  const seg = e.target.closest('.tl-seg');
  if (seg) {
    window.pkTimeline.play(parseFloat(seg.dataset.s), parseFloat(seg.dataset.e));
    return;
  }
  e.preventDefault();
  const a = audio();
  element._dragging = true;
  if (a) a.pause();
  seek(timeAt(t, e.clientX), false);
  const move = (ev) => seek(timeAt(t, ev.clientX), false);
  const up = (ev) => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    element._dragging = false;
    seek(timeAt(t, ev.clientX), true);
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
});

// Space plays / pauses anywhere on the page, as in any media player, unless the user is typing. It
// must win over a focused button: Space would otherwise press it, and "Transcribe" would run the model.
// Buttons fire on keyup, so both events are taken, in the capture phase, before anything else sees them.
if (!window.pkKeysBound) {
  window.pkKeysBound = true;
  const typing = (t) => t && (t.isContentEditable || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT'
    || (t.tagName === 'INPUT' && !['checkbox', 'radio', 'button', 'submit', 'range'].includes(t.type)));
  const onSpace = (e) => {
    if (e.code !== 'Space' || typing(e.target) || !audio()) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'keydown' && !e.repeat) togglePlay();
  };
  document.addEventListener('keydown', onSpace, true);
  document.addEventListener('keyup', onSpace, true);
  document.addEventListener('keydown', (e) => {
    if (typing(e.target) || !tl()) return;
    if (e.key === '?') { e.preventDefault(); toggleHelp(); }
    else if (e.key === 'Escape') toggleHelp(false);
  });
  // ← / → jump 5 s, keeping play / pause as it is; arrows that move through tabs or the table are theirs.
  document.addEventListener('keydown', (e) => {
    if ((e.code !== 'ArrowLeft' && e.code !== 'ArrowRight') || e.altKey || e.ctrlKey || e.metaKey) return;
    if (typing(e.target) || e.target.closest('[role=tab], [role=tablist], [role=grid], table')) return;
    const t = tl(), a = audio();
    if (!t || !a) return;
    e.preventDefault();
    const x = Math.max(0, Math.min(parseFloat(t.dataset.dur), a.currentTime + (e.code === 'ArrowLeft' ? -SKIP : SKIP)));
    a.currentTime = x;
    element._stopAt = null;  // a jump leaves "play just this turn"
    setTime(t, x);
    const sc = t.querySelector('.tl-scroll'), px = lw(t) + x * element._pps;  // keep the playhead in view
    if (px < sc.scrollLeft + lw(t) || px > sc.scrollLeft + sc.clientWidth - 40) sc.scrollLeft = px - lw(t) - sc.clientWidth / 3;
  }, true);
}

element.addEventListener('wheel', (e) => {
  const t = tl();
  if (!t || !(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  const sc = t.querySelector('.tl-scroll');
  zoom(t, e.deltaY < 0 ? 1.25 : 0.8, e.clientX - sc.getBoundingClientRect().left);
}, {passive: false});

watch('value', init);
init();
"""

# For the transcript views: clicking a turn plays it in the timeline's player.
PLAY_ON_CLICK_JS = r"""
// Merging speakers (Per speaker tab): the page keeps {from: into} in a hidden box (#pk-merges) and clicks a
// hidden button to send it; the server redraws the timeline, the views and the downloads. null undoes all.
window.pkMerge = (from, into) => {
  const box = document.querySelector('#pk-merges textarea, #pk-merges input');
  const send = document.getElementById('pk-merges-apply');
  if (!box || !send) return;
  let merges = {};
  try { merges = JSON.parse(box.value || '{}'); } catch (err) { merges = {}; }
  if (from === null) merges = {}; else merges[from] = into;
  box.value = JSON.stringify(merges);
  box.dispatchEvent(new Event('input', {bubbles: true}));
  setTimeout(() => send.click(), 50);
};
element.addEventListener('change', (e) => {
  const select = e.target.closest('.pk-merge');
  if (select && select.value !== '') window.pkMerge(select.dataset.from, select.value);
});
element.addEventListener('click', (e) => {
  if (e.target.closest('.pk-merge')) { e.preventDefault(); e.stopPropagation(); return; }  // not a collapse
  if (e.target.closest('.pk-unmerge')) { window.pkMerge(null); return; }
  const rename = e.target.closest('[data-rename]');
  if (rename) {  // inside a <summary>: rename, without also collapsing the speaker
    e.preventDefault();
    e.stopPropagation();
    if (window.pkRename) window.pkRename(rename.dataset.rename);
    return;
  }
  const row = e.target.closest('[data-s]');
  if (row && window.pkTimeline) window.pkTimeline.play(parseFloat(row.dataset.s), parseFloat(row.dataset.e));
});
// The search box over the Conversation: show only the turns that match, and mark them on the timeline.
element.addEventListener('input', (e) => {
  if (!e.target.matches('.pk-search')) return;
  const q = e.target.value.trim().toLowerCase();
  let n = 0;
  element.querySelectorAll('.pk-turn').forEach((row) => {
    const hit = !q || row.querySelector('.pk-txt').textContent.toLowerCase().includes(q);
    row.classList.toggle('pk-hide', !hit);
    if (q && hit) n++;
  });
  const transcript = element.querySelector('.pk-transcript');
  if (transcript) transcript.classList.toggle('filtering', !!q);
  const count = element.querySelector('.pk-search-count');
  if (count) count.textContent = q ? `${n} turn${n === 1 ? '' : 's'}` : '';
  document.querySelectorAll('.tl').forEach((t) => {
    t.classList.toggle('searching', !!q);
    t.querySelectorAll('.tl-seg:not(.tl-refseg)').forEach((s) =>
      s.classList.toggle('tl-match', !!q && s.textContent.toLowerCase().includes(q)));
  });
});
"""
