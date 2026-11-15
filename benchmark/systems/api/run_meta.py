"""Meta Muse Voice Transcribe (realtime WebSocket, DIARIZATION mode) over the comparison sets.

    python benchmark/systems/api/run_meta.py meetings [--streams 16] [--sessions FILE]

Each recording is streamed at real-time pace (80 ms frames; faster is refused with close 1008), N recordings in
parallel. Recordings longer than the 60 min session cap go as consecutive streams of <= MAX_STREAM_S; a later part's
speaker labels get a `p<k>_` prefix (no cross-session speaker identity). Raw events are kept in `response`.
Output: $BENCH_OUT/meta/<corpus>/<session>.json; key in $API_KEY_DIR/meta.
"""
import argparse, asyncio, json, os, sys, time, uuid

import numpy as np, soundfile as sf, websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meta_events import segments
from run_api import bench, read_key

MAX_STREAM_S, FRAME = 3480.0, 2560  # 58 min; 80 ms of 16 kHz int16


async def stream(pcm, key):
    events = []
    async with websockets.connect(f"wss://api.meta.ai/v1/asr/realtime?sessionId={uuid.uuid4()}", max_size=None,
                                  ping_interval=20, ping_timeout=60) as ws:
        await ws.send(json.dumps({"authorization": {"accessToken": f"Bearer {key}"}, "audioEncoding": "PCM_16KHZ",
                                  "model": "muse-voice-transcribe-1.0", "mode": "DIARIZATION",
                                  "emitAudioProgress": False}))
        hs = json.loads(await ws.recv())
        if "sessionId" not in hs:
            raise RuntimeError(f"handshake failed: {hs}")
        closed = {}

        async def rx():
            try:
                async for m in ws:
                    if isinstance(m, str):
                        events.append(json.loads(m))
            except websockets.ConnectionClosed as e:
                closed.update(code=e.code, reason=e.reason)
        task = asyncio.create_task(rx())
        start = time.time()
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME])
            lag = start + (i + FRAME) / 32000 - time.time()
            if lag > 0:
                await asyncio.sleep(lag)
        await ws.send(json.dumps({"type": "endStream"}))
        await task
    code = closed.get("code", ws.close_code)
    if code not in (None, 1000):
        raise RuntimeError(f"closed {code}: {closed.get('reason') or ws.close_reason}")
    errs = [e for e in events if e.get("type") == "error"]
    if errs:
        raise RuntimeError(f"server error: {errs[0]}")
    return events


async def run_one(sem, key, out, dur, c, sess, path):
    async with sem:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(1)
        if sr != 16000:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
        n = int(np.ceil(dur / MAX_STREAM_S))
        step = int(np.ceil(len(pcm) / 2 / n)) * 2
        t0, segs, raw, err = time.time(), [], [], None
        for k in range(n):
            off = k * step / 32000
            for attempt in range(5):
                try:
                    ev = await stream(pcm[k * step:(k + 1) * step], key)
                    break
                except Exception as e:
                    err = repr(e)
                    print(f"  {c} {sess} part {k} attempt {attempt + 1}: {err[:200]}", flush=True)
                    if "1008" in err or "handshake failed" in err:
                        break
                    await asyncio.sleep(30 * (attempt + 1))
            else:
                ev = None
            if ev is None:
                break
            err = None
            raw.append(ev)
            segs += [{**g, "start": g["start"] + off, "end": g["end"] + off,
                      "speaker": g["speaker"] if k == 0 else f"p{k}_{g['speaker']}"} for g in segments(ev)]
        os.makedirs(f"{out}/{c}", exist_ok=True)
        json.dump({"session": sess, "corpus": c, "duration": dur, "decode_s": time.time() - t0, "error": err,
                   "raw": "", "segments": segs, "response": raw, "parts": n},
                  open(f"{out}/{c}/{sess}.json", "w"))
        print(f"{c} {sess} {dur / 60:.1f} min, {n} stream(s), {len(segs)} segs{', ERROR ' + err[:100] if err else ''}",
              flush=True)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out", default=bench.system_dir("meta"))
    p.add_argument("--streams", type=int, default=16)
    p.add_argument("--sessions")
    a = p.parse_args()
    key = read_key("meta")
    jobs = bench.jobs(a.corpora, a.out, a.sessions)  # longest first: they bound the wall clock
    print(f"{len(jobs)} sessions, {sum(j[0] for j in jobs) / 3600:.2f} h, {a.streams} parallel streams", flush=True)
    sem = asyncio.Semaphore(a.streams)
    await asyncio.gather(*(run_one(sem, key, a.out, *j) for j in jobs))


if __name__ == "__main__":
    asyncio.run(main())
