"""The Hugging Face revision of every model behind the published numbers, and a way to run at exactly those.

    python benchmark/models.py --fetch [REPO ...]      # all, or only the named repos
    export HF_HUB_OFFLINE=1                            # then run the systems as usual

--fetch downloads each pinned revision into the Hugging Face cache and points the repo's `main` ref at it, so with
HF_HUB_OFFLINE=1 every loader (transformers, NeMo, DiariZen, the demo) resolves the model name to the pinned
snapshot instead of whatever is upstream now. Revisions marked (date) were not recorded at run time and are the
repository's last commit before that run.
"""
import argparse
import os

MODELS = {
    # Pandemonium (all conditions) and the released pipeline
    "BUT-FIT/Pandemonium": "fa9d16b0bf1983dd6e13e874cdd4c9a0e62cf19b",
    "nvidia/Nemotron-3-Diarization": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
    "nvidia/parakeet-tdt-0.6b-v3": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",  # tokenizer fallback, single-speaker
    # DiariZen and the systems decoded on it
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2": "027b3e221b9d81dce2be794084f1dbd3ba2da403",  # (2026-10-23)
    "pyannote/wespeaker-voxceleb-resnet34-LM": "837717ddb9ff5507820346191109dc79c958d614",
    "BUT-FIT/SE-DiCoW": "470fce9ffff844dd53a27751cdf6c6df9efecb39",
    "BUT-FIT/DiCoW_v3_3": "c34b64d9a9c5148c65fd355bb188d60343a6b44f",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "Ease3/Xiaomi-CocktailASR-1": "b99aafe1d6c83ee1d95a596acdb2f256bc17e1ca",  # (2026-10-25)
    # other open models
    "nvidia/multitalker-parakeet-streaming-0.6b-v1": "8749fc71fd6e2d88ef230159bbf2aea69b524ee1",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize": "704aa4a9c304e8520be88901e0d1960158ef5b15",  # (2026-10-25)
    "microsoft/VibeVoice-ASR": "d0c9efdb8d614685062c04425d91e01b6f37d944",  # (2026-10-25)
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B": "9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341",  # ~60 GB
}


def fetch(repo, revision):
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    snapshot_download(repo, revision=revision)
    ref = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "refs", "main")
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(revision)
    print(f"{repo} @ {revision[:10]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", nargs="*", metavar="REPO")
    a = ap.parse_args()
    if a.fetch is None:
        for repo, rev in MODELS.items():
            print(f"{repo:48s} {rev}")
        return
    for repo in a.fetch or MODELS:
        fetch(repo, MODELS[repo])


if __name__ == "__main__":
    main()
