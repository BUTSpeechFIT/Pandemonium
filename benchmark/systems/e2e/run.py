"""Run an end-to-end diarizing ASR model (MOSS-Transcribe-Diarize or VibeVoice-ASR) over the comparison sets.

    python benchmark/systems/e2e/run.py {moss,vibevoice} meetings ihm_mix [--shard K --nshards N] [--sessions FILE]
    python benchmark/systems/e2e/run.py vibevoice mixtures --max-new-tokens 4096

Official greedy HF-transformers inference, batch 1, whole recordings. Each model runs in its own environment (see
setup.sh); VibeVoice also needs its repository's demo/ on the path (VIBEVOICE_REPO).

One JSON per session in $BENCH_OUT/<model>/<corpus>/<session>.json: raw text, parsed segments
[{start, end, speaker, text}], audio duration, decode seconds (model.generate incl. audio frontend,
excluding model load). Sessions are split across shards by LPT on duration; done files are skipped.
"""
import argparse, json, os, sys, time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402


class Moss:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
        mid = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, dtype="auto",
                                                          attn_implementation=attn).to(torch.bfloat16).cuda().eval()
        self.proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self.parse, self.msgs, self.gen = parse_transcript, build_transcription_messages, generate_transcription

    def __call__(self, path):
        r = self.gen(self.model, self.proc, self.msgs(path), max_new_tokens=65536, do_sample=False,
                     device=torch.device("cuda"), dtype=torch.bfloat16)
        segs = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in self.parse(r["text"])]
        return r["text"], segs


class VibeVoice:
    def __init__(self, max_new_tokens=32768):
        self.max_new_tokens = max_new_tokens
        sys.path.insert(0, os.path.join(os.environ.get("VIBEVOICE_REPO", "VibeVoice"), "demo"))
        from vibevoice_asr_inference_from_file import VibeVoiceASRBatchInference
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.inf = VibeVoiceASRBatchInference("microsoft/VibeVoice-ASR", device="cuda", attn_implementation=attn)

    def __call__(self, path):
        r = self.inf.transcribe_batch([path], max_new_tokens=self.max_new_tokens, do_sample=False)[0]
        segs = [{"start": float(s.get("start_time", 0)), "end": float(s.get("end_time", 0)),
                 "speaker": str(s.get("speaker_id", "")), "text": s.get("text", "")} for s in r["segments"]]
        return r["raw_text"], segs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["moss", "vibevoice"])
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--sessions", help="JSON {corpus: [session, ...]} to restrict to (e.g. the RTFx subset)")
    p.add_argument("--warmup", help="audio file decoded once before timing starts")
    p.add_argument("--max-new-tokens", type=int, default=32768,
                   help="VibeVoice generation cap; 4096 on the short mixtures, where real outputs are <400 tokens and "
                        "loops otherwise run 25 min to 32768")
    p.add_argument("--reverse", action="store_true",
                   help="shortest first: a helper started later meets the running shards only at the end")
    a = p.parse_args()
    a.out = a.out or bench.system_dir(a.model)

    todo = bench.jobs(a.corpora, a.out, a.sessions, a.shard, a.nshards)
    if a.reverse:
        todo.reverse()
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} to do, {sum(j[0] for j in todo) / 3600:.2f} h", flush=True)
    if not todo:
        return

    model = Moss() if a.model == "moss" else VibeVoice(a.max_new_tokens)
    if a.warmup:
        model(a.warmup)
    for dur, c, sess, path in todo:
        if os.path.exists(f"{a.out}/{c}/{sess}.json"):  # done meanwhile by another worker
            continue
        os.makedirs(f"{a.out}/{c}", exist_ok=True)
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            raw, segs = model(path)
            err = None
        except Exception as e:  # keep going; a failed session is recorded, not silently dropped
            raw, segs, err = "", [], repr(e)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dt = time.time() - t0
        json.dump({"session": sess, "corpus": c, "duration": dur, "decode_s": dt, "error": err,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "raw": raw, "segments": segs},
                  open(f"{a.out}/{c}/{sess}.json", "w"))
        print(f"{c} {sess} {dur / 60:.1f} min -> {dt:.1f} s ({dur / dt:.1f}x), {len(segs)} segs"
              f"{', ERROR ' + err if err else ''}", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
