"""The GPU half of the demo, on another machine: started over SSH by `remote.py`, one job at a time.

The demo server keeps one SSH connection per remote GPU open; the connection's forced command
(deploy/run_worker.sh) runs this. Requests and replies are frames on stdin / stdout:

    4-byte big-endian length, a JSON header of that length, then `header["nbytes"]` raw bytes.

No pickle: what arrives is numbers and JSON, so a compromised demo server can ask this machine to
run the two models and nothing else. Anything a library prints goes to stderr, never into the replies.

Requests (`audio` is an id of the recording; its float32 samples come along only when this worker
does not hold that recording already, otherwise it answers {"ok": false, "error": "need-audio"}):

    {"op": "hello"}
    {"op": "diarize", "audio": id, "nbytes": n}                         + audio
    {"op": "transcribe", "audio": id, "audio_bytes": a, "nbytes": a + p,
     "shape": [G, T], "frame_shift": f, "segments": [[s, e, k], ...]}  + audio + probs
"""

import json
import os
import socket
import struct
import sys
import time

import numpy as np

# Before anything imports torch / NeMo: the replies get their own copy of stdout, and fd 1 (where
# libraries write) becomes stderr.
_out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr
_in = os.fdopen(os.dup(0), "rb", buffering=0)


def read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def receive():
    header = json.loads(read_exact(struct.unpack(">I", read_exact(4))[0]))
    return header, read_exact(int(header.get("nbytes", 0)))


def reply(header: dict, payload: bytes = b"") -> None:
    header = dict(header, nbytes=len(payload))
    data = json.dumps(header).encode()
    _out.write(struct.pack(">I", len(data)) + data + payload)


def main() -> None:
    from pipeline import DemoPipeline, Diarization

    started = time.time()
    pipeline = DemoPipeline()
    pipeline.warm_up()
    import torch

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"worker ready on {socket.gethostname()} ({gpu}) in {time.time() - started:.0f} s", file=sys.stderr,
          flush=True)
    held_id, held = None, None  # the last recording, so transcribing it after diarizing sends it once

    while True:
        try:
            header, payload = receive()
        except EOFError:
            return  # the demo server closed the connection
        op = header.get("op")
        try:
            if op == "hello":
                reply({"ok": True, "host": socket.gethostname(), "gpu": gpu})
                continue
            audio_bytes = int(header.get("audio_bytes", header.get("nbytes", 0) if op == "diarize" else 0))
            if audio_bytes:
                held_id, held = header["audio"], np.frombuffer(payload[:audio_bytes], dtype=np.float32).copy()
            elif header.get("audio") != held_id:
                reply({"ok": False, "error": "need-audio"})
                continue
            if op == "diarize":
                diar = pipeline.diarize(held)
                probs = np.ascontiguousarray(diar.probs, dtype=np.float32)
                reply({"ok": True, "shape": list(probs.shape), "frame_shift": diar.frame_shift,
                       "segments": [list(s) for s in diar.segments]}, probs.tobytes())
            elif op == "transcribe":
                probs = np.frombuffer(payload[audio_bytes:], dtype=np.float32).reshape(header["shape"])
                diar = Diarization(probs=probs.copy(), frame_shift=float(header["frame_shift"]),
                                   segments=[(float(s), float(e), int(k)) for s, e, k in header["segments"]])
                words = pipeline.transcribe(held, diar, soft=True)
                reply({"ok": True, "words": [[[w.text, w.start, w.end] for w in row] for row in words]})
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # one failed job must not take the worker down
            print(f"worker: {op} failed: {e!r}", file=sys.stderr, flush=True)
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
