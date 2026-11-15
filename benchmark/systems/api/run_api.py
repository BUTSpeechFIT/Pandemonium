"""Run a hosted diarizing ASR API over the comparison sets; same output layout as run_e2e.py.

    python benchmark/systems/api/run_api.py PROVIDER meetings [--sessions FILE] [--workers 4]

Audio is sent as 16 kHz mono FLAC (lossless, ~1/3 of the WAV). OUT_DIR/<corpus>/<session>.json holds the parsed
segments plus the provider's full response (`response`), and `decode_s` = request wall time (upload included).
Output: $BENCH_OUT/<provider>/<corpus>/<session>.json. Keys are read from $API_KEY_DIR/<provider> (default
~/.config/api-asr) and never printed. Settings behind the published rows (see benchmark/README.md):
  gemini   GEMINI_WIN_S=900 (15 min windows, Gemini's own speakers per window, linked through DiariZen)
  azure    AZURE_LOCALE=en for the sessions it otherwise returns with every timestamp at 0
  smallest SMALLEST_MODEL=pulse (pulse-pro ignores diarize)
"""
import argparse, io, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402
lock = threading.Lock()
out_of_credit = threading.Event()  # set on a quota/billing refusal: stop sending, nothing more can succeed


def flac(path):
    import librosa
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != 16000:
        a, sr = librosa.resample(a, orig_sr=sr, target_sr=16000), 16000
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="FLAC")
    return buf.getvalue()


def mistral(path, key):
    r = requests.post("https://api.mistral.ai/v1/audio/transcriptions",
                      headers={"Authorization": f"Bearer {key}"},
                      files={"file": (os.path.basename(path).rsplit(".", 1)[0] + ".flac", flac(path), "audio/flac")},
                      data={"model": "voxtral-mini-latest", "diarize": "true", "timestamp_granularities": "segment"},
                      timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    segs = [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
             "speaker": str(s.get("speaker_id", s.get("speaker", "")) or ""), "text": s.get("text", "")}
            for s in resp.get("segments", [])]
    return resp, segs


def elevenlabs(path, key):
    r = requests.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": key},
                      files={"file": (os.path.basename(path).rsplit(".", 1)[0] + ".flac", flac(path), "audio/flac")},
                      data={"model_id": "scribe_v2", "diarize": "true", "timestamps_granularity": "word",
                            "tag_audio_events": "false"},
                      timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    segs = []  # words -> segments: a new one at each speaker change or pause > 1 s
    for w in resp.get("words", []):
        if w.get("type") != "word" or w.get("start") is None:
            continue
        spk = str(w.get("speaker_id") or "")
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] <= 1.0:
            segs[-1]["text"] += " " + w["text"]
            segs[-1]["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["text"]})
    return resp, segs


OPENAI_CHUNK_S = 1200  # the diarize model takes ~20 min per request; longer recordings are cut into equal chunks


def _mp3(audio, sr=16000):
    """96 kbps mono MP3: the endpoint takes no FLAC and caps uploads at 25 MB (a 20 min WAV is 38 MB)."""
    import subprocess
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                           "-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3", "-"],
                          input=audio.astype("float32").tobytes(), capture_output=True, check=True).stdout


def _openai_request(key, audio, refs):
    import base64
    data = [("model", "gpt-4o-transcribe-diarize"), ("response_format", "diarized_json"), ("chunking_strategy", "auto")]
    for name, clip in refs.items():
        data += [("known_speaker_names[]", name),
                 ("known_speaker_references[]", "data:audio/mpeg;base64," + base64.b64encode(_mp3(clip)).decode())]
    r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                      files={"file": ("audio.mp3", _mp3(audio), "audio/mpeg")}, data=data, timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def openai(path, key):
    """Speaker labels restart in every request, so each chunk after the first is given 2-10 s clean clips of up to
    four speakers already heard (the API's known-speaker references; four is its maximum), chosen by most speech
    so far. A speaker the model does not match keeps a chunk-local label; recordings of one chunk are unaffected."""
    import librosa
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
    dur = len(audio) / sr
    n = int(np.ceil(dur / OPENAI_CHUNK_S))
    bounds = [(i * dur / n, (i + 1) * dur / n) for i in range(n)]
    segs, resps, refs, talk = [], [], {}, {}
    cache = bench.system_dir("openai_cache")  # chunk responses, so a run stopped by credits resumes for free
    os.makedirs(cache, exist_ok=True)
    for k, (s, e) in enumerate(bounds):
        # a chunk is paid for once: a later chunk failing must not throw the earlier responses away
        cf = os.path.join(cache, f"{os.path.basename(path)}.{k}.{'+'.join(sorted(refs)) or 'none'}.json")
        if os.path.exists(cf):
            resp = json.load(open(cf))
        else:
            resp = _openai_request(key, audio[int(s * sr):int(e * sr)], refs)
            json.dump(resp, open(cf, "w"))
        resps.append(resp)
        chunk = []
        for g in resp.get("segments", []):
            spk = str(g.get("speaker", ""))
            spk = spk if spk in refs else f"c{k}_{spk}"
            chunk.append({"start": g["start"] + s, "end": g["end"] + s, "speaker": spk, "text": g.get("text", "")})
        segs += chunk
        for g in chunk:
            talk[g["speaker"]] = talk.get(g["speaker"], 0.0) + g["end"] - g["start"]
        for spk in sorted(talk, key=talk.get, reverse=True):  # clean clip for the most talkative unenrolled speakers
            if len(refs) >= 4:
                break
            if spk in refs:
                continue
            others = [(g["start"], g["end"]) for g in segs if g["speaker"] != spk]
            clean = [g for g in chunk if g["speaker"] == spk and g["end"] - g["start"] >= 2.0
                     and not any(a < g["end"] and b > g["start"] for a, b in others)]
            if clean:
                g = max(clean, key=lambda g: g["end"] - g["start"])
                # the API wants 1.2-10 s and MP3 encoding pads a few ms, so a 10.0 s cut is rejected as too long
                refs[spk] = audio[int(g["start"] * sr):int(min(g["end"], g["start"] + 9.5) * sr)]
    return {"chunks": bounds, "known_speakers": list(refs), "responses": resps}, segs


# Gemini's word timings drift past ~30-60 s, so by default it gets ~1 min windows. GEMINI_WIN_S raises it for a
# cpWER-only run, where timing does not matter: a recording that fits one window keeps Gemini's own speakers.
GEMINI_WIN_S = float(os.environ.get("GEMINI_WIN_S", 60))
GEMINI_MIN_S = GEMINI_WIN_S * 2 / 3
GEMINI_MAX_OUT = int(os.environ.get("GEMINI_MAX_OUT", 65536))  # default 8192 truncates ~35 min of meeting
DIARIZEN_SEGMENTS = os.path.join(bench.system_dir("diarizen"), "{}", "segments.json")
_path_index = {}


def _sec(x):
    return float(str(x).rstrip("s"))


def _session_of(path):
    if not _path_index:
        for c in bench.MEETINGS + bench.IHM_MIX + bench.MIXTURES:
            if not os.path.exists(bench.manifest_path(c)):
                continue
            for sess, p in bench.load_manifest(c).items():
                _path_index[p] = (c, sess)
    return _path_index[path]


def _cut_points(dsegs, dur):
    """Window ends near every GEMINI_WIN_S, at the instant inside [prev + MIN, prev + WIN] with the fewest active
    DiariZen speakers (ideally none), nearest the target, on a 0.1 s grid."""
    grid = np.arange(0, dur + 0.1, 0.1)
    active = np.zeros(len(grid), int)
    for a, b, _ in dsegs:
        active[int(a * 10):int(b * 10) + 1] += 1
    cuts, prev = [0.0], 0.0
    while dur - prev > GEMINI_WIN_S:
        lo, hi = int((prev + GEMINI_MIN_S) * 10), int((prev + GEMINI_WIN_S) * 10)
        cand = np.arange(lo, hi + 1)
        best = cand[np.lexsort((-(cand), active[cand]))[0]]  # fewest speakers, then the latest such instant
        prev = grid[best]
        cuts.append(float(prev))
    cuts.append(dur)
    return list(zip(cuts[:-1], cuts[1:]))


def _to_diarizen(words_by_spk, dsegs, lo, hi):
    """Map a window's Gemini speakers onto DiariZen speakers: Hungarian on co-speaking time, then any leftover
    Gemini speaker to the DiariZen speaker it overlaps most. No overlap at all -> window-local label."""
    from scipy.optimize import linear_sum_assignment
    dz = sorted({k for a, b, k in dsegs if a < hi and b > lo})
    gs = sorted(words_by_spk)
    if not dz or not gs:
        return {}
    ov = np.array([[sum(max(0.0, min(we, b) - max(ws, a)) for ws, we in words_by_spk[g] for a, b, k in dsegs if k == d)
                    for d in dz] for g in gs])
    m = {}
    for r, c in zip(*linear_sum_assignment(-ov)):
        if ov[r, c] > 0:
            m[gs[r]] = dz[c]
    for r, g in enumerate(gs):
        if g not in m and ov[r].max() > 0:
            m[g] = dz[int(ov[r].argmax())]
    return m


# AI Studio Tier 1 caps gemini-3.5-transcribe at 10 requests/min per project, retries included; bursting past it
# only buys 429s, so studio requests are spaced evenly across all threads and workers.
_pace_lock, _pace_next = threading.Lock(), [0.0]


def _gemini_window(client, types, clip, sr, pace=0.0):
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="FLAC")
    part = types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/flac")
    if len(buf.getvalue()) > 15e6 and not client.vertexai:  # inline is capped at 20 MB; Files API is Developer-API only
        buf.seek(0)
        f = client.files.upload(file=buf, config=types.UploadFileConfig(mime_type="audio/flac"))
        part = types.Part.from_uri(file_uri=f.uri, mime_type="audio/flac")
    for attempt in range(8):
        if pace:
            with _pace_lock:
                slot = max(time.time(), _pace_next[0])
                _pace_next[0] = slot + pace
            time.sleep(max(0.0, slot - time.time()))
        try:
            r = client.models.generate_content(
                model="gemini-3.5-transcribe",
                contents=[part],
                config=types.GenerateContentConfig(max_output_tokens=GEMINI_MAX_OUT, audio_transcription_config=types.AudioTranscriptionConfig(
                    diarization=True, word_timestamp=True, language_codes=["en-US"])))
            d = r.model_dump(mode="json", exclude_none=True)
        except Exception as ex:
            code = getattr(ex, "code", "")
            if code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise RuntimeError(f"HTTP {code}: {str(ex)[:400]}")
            with lock:
                print(f"  window retry {attempt + 1} after HTTP {code}: {str(ex)[:120]}", flush=True)
            time.sleep(min(15 * (attempt + 1), 120))
            continue
        # the model stops near 8k output tokens whatever max_output_tokens says: a cut-off window must not pass as done
        if ((d.get("candidates") or [{}])[0].get("finish_reason") or "") == "MAX_TOKENS":
            raise RuntimeError("MAX_TOKENS: window transcript truncated, use a shorter GEMINI_WIN_S")
        return d


def gemini(path, key, studio=False):
    """gemini-3.5-transcribe on Vertex (express-mode API key), on ~60 s windows cut at DiariZen pauses; Gemini's
    per-window speakers are mapped onto DiariZen's speakers (the diarization SE-DiCoW and CocktailASR use)."""
    import librosa
    from concurrent.futures import ThreadPoolExecutor as TPE
    from google import genai
    from google.genai import types
    if studio:  # Gemini Developer API (AI Studio prepay): a capacity pool separate from Vertex
        clients = [genai.Client(api_key=key)] * 3
    else:  # global/us/eu are separate capacity pools for this preview model; windows go round-robin across them
        clients = [genai.Client(vertexai=True, api_key=key)] + [
            genai.Client(vertexai=True, api_key=key,
                         http_options=types.HttpOptions(base_url=f"https://aiplatform.{loc}.rep.googleapis.com/"))
            for loc in ("us", "eu")]
    corpus, sess = _session_of(path)
    dsegs = json.load(open(DIARIZEN_SEGMENTS.format(corpus))).get(sess, [])
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
    wins = _cut_points(dsegs, len(audio) / sr)
    with TPE(int(os.environ.get("GEMINI_WINDOW_THREADS", 4))) as ex:
        pace = 60.0 / float(os.environ.get("GEMINI_STUDIO_RPM", 9.5)) if studio else 0.0
        resps = list(ex.map(lambda kw: _gemini_window(clients[kw[0] % 3], types,
                                                      audio[int(kw[1][0] * sr):int(kw[1][1] * sr)], sr, pace), enumerate(wins)))
    segs, maps, usage = [], [], []
    for k, ((lo, hi), d) in enumerate(zip(wins, resps)):
        usage.append(d.get("usage_metadata"))
        turns = []
        for part in (d.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
            t = part.get("audio_transcription") or {}
            # a word occasionally comes without one offset; it takes the other (or 0 s for the first word)
            ws = [(_sec(w.get("start_offset", w.get("end_offset", 0))) + lo,
                   _sec(w.get("end_offset", w.get("start_offset", 0))) + lo) for w in t.get("words") or []]
            if ws:
                turns.append((t.get("speaker_label", ""), ws, t.get("text", "")))
        by_spk = {}
        for g, ws, _ in turns:
            by_spk.setdefault(g, []).extend(ws)
        m = {g: g for g in by_spk} if len(wins) == 1 else _to_diarizen(by_spk, dsegs, lo, hi)
        maps.append(m)
        segs += [{"start": ws[0][0], "end": ws[-1][1], "speaker": m.get(g, f"w{k}_{g}"), "text": txt} for g, ws, txt in turns]
    return {"windows": wins, "speaker_maps": maps, "usage": usage,
            "candidates": [(d.get("candidates") or [{}])[0] for d in resps]}, segs


AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "swedencentral")


def azure(path, key):
    """MAI-Transcribe-2 through Azure Speech fast transcription (enhancedMode), with diarization."""
    definition = {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2", "modelOptions": {"timestamps": "segment"}},
                  "diarization": {"enabled": True}}
    if os.environ.get("AZURE_LOCALE"):  # MAI drops ALL timestamps when it detects several languages in a file
        definition["locales"] = [os.environ["AZURE_LOCALE"]]
    r = requests.post(f"https://{AZURE_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe"
                      "?api-version=2025-10-15", headers={"Ocp-Apim-Subscription-Key": key},
                      files={"audio": (os.path.basename(path).rsplit(".", 1)[0] + ".flac", flac(path), "audio/flac"),
                             "definition": (None, json.dumps(definition))}, timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    segs = [{"start": p["offsetMilliseconds"] / 1000, "end": (p["offsetMilliseconds"] + p["durationMilliseconds"]) / 1000,
             "speaker": str(p.get("speaker", "")), "text": p.get("text", "")} for p in resp.get("phrases", [])]
    return resp, segs


def smallest(path, key):
    """smallest.ai Pulse (diarize is a Pulse feature; Pulse Pro ignores it) with diarization and word timestamps; words -> speaker turns."""
    r = requests.post("https://api.smallest.ai/waves/v1/stt/",
                      params={"model": os.environ.get("SMALLEST_MODEL", "pulse"), "language": "en", "word_timestamps": "true", "diarize": "true"},
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/octet-stream"},
                      data=flac(path), timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    segs = []
    for w in resp.get("words", []):
        spk = str(w.get("speaker", w.get("speaker_id", "")))
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] <= 1.0:
            segs[-1]["text"] += " " + w["word"]
            segs[-1]["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})
    return resp, segs


def deepgram(path, key):
    """Deepgram Nova-3 pre-recorded, diarize + punctuation; per-word speakers -> speaker turns."""
    r = requests.post("https://api.deepgram.com/v1/listen",
                      params={"model": "nova-3", "language": "en", "diarize": "true", "punctuate": "true",
                              "smart_format": "true"},
                      headers={"Authorization": f"Token {key}", "Content-Type": "audio/flac"},
                      data=flac(path), timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    resp = r.json()
    words = resp["results"]["channels"][0]["alternatives"][0].get("words", [])
    segs = []
    for w in words:
        spk, txt = str(w.get("speaker", "")), w.get("punctuated_word", w["word"])
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] <= 1.0:
            segs[-1]["text"] += " " + txt
            segs[-1]["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": txt})
    return {"metadata": resp.get("metadata"), "words": words}, segs


def assemblyai(path, key):
    """AssemblyAI async transcription (default best model) with speaker_labels; utterances are speaker turns."""
    h = {"authorization": key}
    up = requests.post("https://api.assemblyai.com/v2/upload", headers=h, data=flac(path), timeout=3600)
    if up.status_code != 200:
        raise RuntimeError(f"HTTP {up.status_code}: {up.text[:500]}")
    job = requests.post("https://api.assemblyai.com/v2/transcript", headers=h, timeout=60,
                        json={"audio_url": up.json()["upload_url"], "speaker_labels": True, "language_code": "en"})
    if job.status_code != 200:
        raise RuntimeError(f"HTTP {job.status_code}: {job.text[:500]}")
    tid = job.json()["id"]
    while True:
        r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=h, timeout=60)
        resp = r.json()
        if resp.get("status") == "completed":
            break
        if resp.get("status") == "error":
            raise RuntimeError(f"job error: {resp.get('error')}")
        time.sleep(3)
    segs = []  # from word timings (utterances can span long turns), split at speaker change or pause > 1 s
    for u in resp.get("utterances") or []:
        for w in u.get("words") or []:
            spk, st, en = str(w.get("speaker", u.get("speaker", ""))), w["start"] / 1000, w["end"] / 1000
            if segs and segs[-1]["speaker"] == spk and st - segs[-1]["end"] <= 1.0:
                segs[-1]["text"] += " " + w["text"]
                segs[-1]["end"] = en
            else:
                segs.append({"start": st, "end": en, "speaker": spk, "text": w["text"]})
    return {k: resp.get(k) for k in ("id", "speech_model", "speech_model_used", "audio_duration", "utterances")}, segs


PROVIDERS = {"assemblyai": assemblyai, "deepgram": deepgram, "smallest": smallest, "azure": azure, "mistral": mistral, "elevenlabs": elevenlabs, "openai": openai, "gemini": gemini,
             "gemini-studio": lambda path, key: gemini(path, key, studio=True)}


def read_key(provider):
    return open(os.path.join(os.path.expanduser(os.environ.get("API_KEY_DIR", "~/.config/api-asr")), provider)).read().strip()


def run_one(provider, key, out, dur, c, sess, path):
    if out_of_credit.is_set():
        return
    # several runners (e.g. Vertex and AI Studio) may share one OUT: each session is claimed by exactly one of them
    os.makedirs(f"{out}/{c}", exist_ok=True)
    try:
        os.close(os.open(f"{out}/{c}/{sess}.claim", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return
    if os.path.exists(f"{out}/{c}/{sess}.json"):
        return
    for attempt in range(12):
        t0 = time.time()
        try:
            resp, segs = PROVIDERS[provider](path, key)
            err = None
            break
        except Exception as e:
            resp, segs, err = None, [], repr(e)
            transient = any(x in err for x in ("HTTP 429", "HTTP 5", "Timeout", "Connection"))
            with lock:
                print(f"  {c} {sess}: attempt {attempt + 1} failed: {err[:200]}", flush=True)
            # Gemini's 429 RESOURCE_EXHAUSTED says "quota" (and "billing" on AI Studio) but is a rate limit, not a stop
            if not ("HTTP 429" in err and "RESOURCE_EXHAUSTED" in err) and any(x in err.lower() for x in ("quota", "credit", "insufficient", "payment", "billing")):
                out_of_credit.set()
                with lock:
                    print(f"  OUT OF CREDIT at {c} {sess}: stopping, this session is not saved", flush=True)
                os.remove(f"{out}/{c}/{sess}.claim")  # unclaimed again, so the next run picks it up
                return
            if not transient:
                break
            time.sleep(min(60 * (attempt + 1), 300))
    dt = time.time() - t0
    os.makedirs(f"{out}/{c}", exist_ok=True)
    json.dump({"session": sess, "corpus": c, "duration": dur, "decode_s": dt, "error": err, "raw": "",
               "segments": segs, "response": resp}, open(f"{out}/{c}/{sess}.json", "w"))
    with lock:
        print(f"{c} {sess} {dur / 60:.1f} min -> {dt:.1f} s, {len(segs)} segs{', ERROR' if err else ''}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("provider", choices=list(PROVIDERS))
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out")
    p.add_argument("--sessions")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.out = a.out or bench.system_dir(a.provider)
    key = read_key(a.provider)
    jobs = bench.jobs(a.corpora, a.out, a.sessions)
    jobs.sort()  # shortest first: long requests hit backend_out_of_capacity most, retry them last
    print(f"{len(jobs)} sessions, {sum(j[0] for j in jobs) / 3600:.2f} h to do", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for j in jobs:
            ex.submit(run_one, a.provider, key, a.out, *j)


if __name__ == "__main__":
    main()
