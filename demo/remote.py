"""Overflow to other GPUs: when this machine's GPU is busy, a job runs on a remote one instead.

Each remote GPU is one long-lived SSH connection to a worker (`worker.py`) that keeps both models
loaded. The local GPU is always tried first; a remote is used only while the local one is taken, and
a job whose remote fails (connection lost, worker crashed, timeout) is run again on the local GPU.

Configured by `DEMO_REMOTE_GPUS` (comma-separated `user@host`), else the lines of
~/.config/pandemonium-demo/remote_gpus. The key (`worker_key`) is accepted on the remote host only
with a forced command that starts the worker (deploy/run_worker.sh), so it opens nothing else there.
"""

import hashlib
import json
import os
import select
import shlex
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import List, Optional

import numpy as np

from pandemonium import SAMPLE_RATE, Word
from pipeline import Diarization

CONFIG_DIR = os.path.expanduser("~/.config/pandemonium-demo")
KEY = os.environ.get("DEMO_WORKER_SSH_KEY", os.path.join(CONFIG_DIR, "worker_key"))
KNOWN_HOSTS = os.environ.get("DEMO_WORKER_KNOWN_HOSTS", os.path.join(CONFIG_DIR, "known_hosts"))
RETRY_AFTER = 60.0  # s before reconnecting to a remote that failed
STARTUP_TIMEOUT = 600.0  # loading both models and warming up, on a cold cache


class RemoteFailed(Exception):
    pass


def configured_remotes() -> List[str]:
    value = os.environ.get("DEMO_REMOTE_GPUS")
    if value is None:
        path = os.path.join(CONFIG_DIR, "remote_gpus")
        value = open(path).read().replace("\n", ",") if os.path.exists(path) else ""
    return [x.strip() for x in value.split(",") if x.strip() and not x.strip().startswith("#")]


def _timeout(audio: np.ndarray) -> float:
    """Generous: ~40x slower than the slowest GPU measured, so only a hung worker reaches it."""
    return 120.0 + len(audio) / SAMPLE_RATE


class LocalGPU:
    name = "local"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def ready(self) -> bool:
        return True

    def diarize(self, audio):
        return self.pipeline.diarize(audio)

    def transcribe(self, audio, diar):
        return self.pipeline.transcribe(audio, diar, soft=True)


class RemoteGPU:
    """One worker over SSH. Not thread-safe by itself: the pool hands it to one job at a time."""

    def __init__(self, target: str):
        self.target = target
        self.name = target.split("@")[-1].split(".")[0]
        self._proc: Optional[subprocess.Popen] = None
        self._held = None  # id of the recording the worker holds
        self.ready = False
        self._lock = threading.Lock()
        threading.Thread(target=self._connect_loop, daemon=True, name=f"remote-{self.name}").start()

    # --- connection --------------------------------------------------------------------------
    def _command(self) -> List[str]:
        if self.target.startswith("exec:"):  # a local command speaking the same protocol (testing)
            return shlex.split(self.target[5:])
        return ["ssh", "-T", "-F", "/dev/null", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
                self.target]

    def _connect_loop(self) -> None:
        while True:
            if not self.ready:
                try:
                    self._start()
                except Exception as e:
                    print(f"remote {self.name}: not available ({e})", file=sys.stderr, flush=True)
                    self._stop()
            time.sleep(RETRY_AFTER)

    def _start(self) -> None:
        with self._lock:
            self._stop()
            started = time.time()
            self._proc = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=sys.stderr, bufsize=0)
            hello = self._call({"op": "hello"}, timeout=STARTUP_TIMEOUT)[0]
            self._held = None
            self.ready = True
        print(f"remote {self.name}: ready, {hello.get('gpu')} on {hello.get('host')} "
              f"({time.time() - started:.0f} s)", file=sys.stderr, flush=True)

    def _stop(self) -> None:
        self.ready = False
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def fail(self, why: str) -> None:
        print(f"remote {self.name}: failed ({why}); retrying in {RETRY_AFTER:.0f} s", file=sys.stderr, flush=True)
        self._stop()

    # --- framing ------------------------------------------------------------------------------
    def _read(self, n: int, deadline: float) -> bytes:
        out, fd = bytearray(), self._proc.stdout.fileno()
        while len(out) < n:
            left = deadline - time.time()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise RemoteFailed("timed out")
            chunk = os.read(fd, min(n - len(out), 1 << 20))
            if not chunk:
                raise RemoteFailed("connection closed")
            out += chunk
        return bytes(out)

    def _call(self, header: dict, payload: bytes = b"", timeout: float = 120.0):
        if self._proc is None or self._proc.poll() is not None:
            raise RemoteFailed("not connected")
        data = json.dumps(dict(header, nbytes=len(payload))).encode()
        try:
            self._proc.stdin.write(struct.pack(">I", len(data)) + data + payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RemoteFailed(f"send: {e}")
        deadline = time.time() + timeout
        reply = json.loads(self._read(struct.unpack(">I", self._read(4, deadline))[0], deadline))
        return reply, self._read(int(reply.get("nbytes", 0)), deadline)

    def _job(self, header: dict, audio: np.ndarray, extra: bytes = b""):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        audio_id = hashlib.sha1(audio.tobytes()).hexdigest()
        with self._lock:
            for send_audio in ((False, True) if audio_id == self._held else (True,)):
                samples = audio.tobytes() if send_audio else b""
                reply, payload = self._call(dict(header, audio=audio_id, audio_bytes=len(samples)),
                                            samples + extra, timeout=_timeout(audio))
                if reply.get("error") == "need-audio":
                    continue
                self._held = audio_id
                if not reply.get("ok"):
                    raise RemoteFailed(reply.get("error", "unknown error"))
                return reply, payload
            raise RemoteFailed("worker kept asking for the audio")

    # --- the pipeline's two calls -------------------------------------------------------------
    def diarize(self, audio: np.ndarray) -> Diarization:
        reply, payload = self._job({"op": "diarize"}, audio)
        probs = np.frombuffer(payload, dtype=np.float32).reshape(reply["shape"]).copy()
        return Diarization(probs=probs, frame_shift=float(reply["frame_shift"]),
                          segments=[(float(s), float(e), int(k)) for s, e, k in reply["segments"]])

    def transcribe(self, audio: np.ndarray, diar: Diarization) -> List[List[Word]]:
        probs = np.ascontiguousarray(diar.probs, dtype=np.float32)
        reply, _ = self._job({"op": "transcribe", "shape": list(probs.shape), "frame_shift": diar.frame_shift,
                              "segments": [list(s) for s in diar.segments]}, audio, probs.tobytes())
        return [[Word(t, float(s), float(e)) for t, s, e in row] for row in reply["words"]]


class GPUPool:
    """Hands each job a GPU: the local one if free, else a ready remote one, else waits for either."""

    def __init__(self, pipeline, remotes: List[str]):
        self.local = LocalGPU(pipeline)
        self.remotes = [RemoteGPU(r) for r in remotes]
        self._busy = set()
        self._cond = threading.Condition()

    @property
    def size(self) -> int:
        """How many jobs can run at once, at most (remotes that are down make it fewer)."""
        return 1 + len(self.remotes)

    def _acquire(self, local_only: bool = False):
        with self._cond:
            while True:
                for gpu in [self.local] + ([] if local_only else self.remotes):
                    if gpu not in self._busy and gpu.ready:
                        self._busy.add(gpu)
                        return gpu
                self._cond.wait(timeout=5)  # also notices a remote coming back

    def _release(self, gpu) -> None:
        with self._cond:
            self._busy.discard(gpu)
            self._cond.notify_all()

    @contextmanager
    def lease(self):
        """`with pool.lease() as gpu: gpu.diarize(...); gpu.transcribe(...)`, on one machine."""
        lease = _Lease(self)
        try:
            yield lease
        finally:
            lease.release()


class _Lease:
    """A GPU for one job; if its remote fails, the job's calls continue on the local GPU."""

    def __init__(self, pool: GPUPool):
        self.pool = pool
        self.gpu = pool._acquire()

    @property
    def where(self) -> str:
        return self.gpu.name

    def _run(self, method: str, *args):
        try:
            return getattr(self.gpu, method)(*args)
        except RemoteFailed as e:
            self.gpu.fail(str(e))
            self.pool._release(self.gpu)
            self.gpu = None
            self.gpu = self.pool._acquire(local_only=True)
            return getattr(self.gpu, method)(*args)

    def diarize(self, audio):
        return self._run("diarize", audio)

    def transcribe(self, audio, diar):
        return self._run("transcribe", audio, diar)

    def release(self) -> None:
        if self.gpu is not None:
            self.pool._release(self.gpu)
            self.gpu = None
