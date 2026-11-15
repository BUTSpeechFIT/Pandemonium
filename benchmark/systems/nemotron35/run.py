"""NVIDIA-Nemotron-3.5-Transcribe-30B-A4B on the MT-ASR sets, through local vLLM servers.

    python benchmark/systems/nemotron35/run.py meetings --ports 8001 8002 ... [--per-server 4] [--sessions FILE]

Full sessions, the model card's exact "Multi-Speaker Transcription" benchmark prompt, temperature 0, reasoning off,
the card's generation-loop guardrail, and CTC word timestamps (the vLLM timestamp middleware). Writes the same
$BENCH_OUT/nemotron35/<corpus>/<session>.json. Start one server per GPU with serve.sh first. 4 requests per server
fit a 96 GB card; 8 ran out of memory on the hour-long sessions.
"""
import argparse, base64, io, json, os, queue, re, sys, threading, time
from pathlib import Path

import soundfile as sf
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
PROMPT = ("Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no "
          "punctuation and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, "
          "<spk:2>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the "
          "speaker-tagged transcript.")

CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)
lock = threading.Lock()


def cleanup_generation_loops(text):
    """The card's guardrail (adapted from Qwen3-ASR, Apache-2.0), applied before scoring as the card does."""
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def wav_b64(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def request(client, path):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64(path)}"}}]}],
        temperature=0, top_p=1.0, max_tokens=40000,  # a dense hour needs ~20k
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False,
                    "spaces_between_special_tokens": False, "seed": 0,
                    "mm_processor_kwargs": {"capture_ctc_timestamps": True}},
        timeout=7200,
    ).model_dump()


def segments(words, max_gap=1.0):
    """Word timestamps -> segments, split at a speaker change or a pause (as for AssemblyAI)."""
    segs = []
    for w in words:
        spk, st, en = str(w.get("speaker")), float(w["start"]), float(w["end"])
        if segs and segs[-1]["speaker"] == spk and st - segs[-1]["end"] <= max_gap:
            segs[-1]["text"] += " " + w["word"]
            segs[-1]["end"] = max(segs[-1]["end"], en)
        else:
            segs.append({"start": st, "end": en, "speaker": spk, "text": w["word"]})
    return segs


def text_segments(text, dur):
    """Fallback when alignment failed: one whole-session segment per speaker tag (cpWER-valid only)."""
    by = {}
    for spk, chunk in re.findall(r"<spk:(\d+)>(.*?)(?=<spk:\d+>|$)", text, re.DOTALL):
        by.setdefault(spk, []).append(chunk.strip())
    return [{"start": 0.0, "end": dur, "speaker": s, "text": " ".join(t)} for s, t in by.items()]


def worker(port, jobs, out):
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
    while True:
        try:
            dur, c, sess, path = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        resp, segs, err, raw = None, [], None, ""
        try:
            resp = request(client, path)
            raw = resp["choices"][0]["message"]["content"] or ""
            clean = cleanup_generation_loops(raw)
            ts = resp.get("ctc_timestamps") or {}
            if ts.get("error") is None and ts.get("words") and clean == raw:
                segs = segments(ts["words"])
            else:
                err = f"alignment: {ts.get('error')}" if ts.get("error") else ("loop" if clean != raw else "no timestamps")
                segs = text_segments(clean, dur)
        except Exception as e:
            err = repr(e)
        dt = time.time() - t0
        os.makedirs(f"{out}/{c}", exist_ok=True)
        json.dump({"session": sess, "corpus": c, "duration": dur, "decode_s": dt, "error": err, "raw": "",
                   "text": raw, "segments": segs, "response": resp}, open(f"{out}/{c}/{sess}.json", "w"))
        with lock:
            print(f"{port} {c} {sess} {dur / 60:.1f} min -> {dt:.1f} s, {len(segs)} segs"
                  f"{', ' + err[:120] if err else ''}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out", default=bench.system_dir("nemotron35"))
    p.add_argument("--ports", nargs="+", type=int, required=True)
    p.add_argument("--per-server", type=int, default=4)
    p.add_argument("--sessions")
    a = p.parse_args()
    todo = bench.jobs(a.corpora, a.out, a.sessions)  # longest first, so the workers finish together
    jobs = queue.Queue()
    for j in todo:
        jobs.put(j)
    print(f"{len(todo)} sessions, {sum(j[0] for j in todo) / 3600:.2f} h to do", flush=True)
    threads = [threading.Thread(target=worker, args=(port, jobs, a.out)) for port in a.ports for _ in range(a.per_server)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
