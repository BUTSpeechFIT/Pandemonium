"""Markdown tables from the systems' scores.json (written by score.py), in the layout of the model card.

    python benchmark/tables.py [--card]

--card applies the model card's naming: the ICSI and DiPCo columns become M1 and M2, and systems the card leaves out
are dropped. Cells a system has not been scored on show '–'.
"""
import argparse
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

# (row label, output directory under $BENCH_OUT)
SYSTEMS = [
    ("Pandemonium + oracle activity", "pandemonium_oracle"),
    ("Pandemonium + Nemotron-3-Diarization", "pandemonium_nemotron"),
    ("Pandemonium + pyannoteAI Precision-3", "pandemonium_precision"),
    ("NVIDIA Nemotron-3.5-Transcribe", "nemotron35"),
    ("SE-DiCoW + DiariZen", "se_dicow"),
    ("DiCoW v3.3 + DiariZen", "dicow_v3_3"),
    ("Nemotron-3-Diarization + MT-Parakeet", "mt_parakeet"),
    ("MOSS-Transcribe-Diarize", "moss"),
    ("API A", "api_a"),
    ("API B", "api_b"),
    ("Xiaomi CocktailASR + DiariZen", "cocktail"),
    ("MAI-Transcribe-2", "azure"),
    ("ElevenLabs Scribe v2", "elevenlabs"),
    ("API C", "api_c"),
    ("VibeVoice-ASR", "vibevoice"),
    ("Gemini 3.5 Transcribe", "gemini"),
    ("API D", "api_d"),
    ("Mistral Voxtral", "mistral"),
    ("OpenAI gpt-4o-transcribe-diarize", "openai"),
]
CARD_DROP = {"nemotron35"}
CARD_MEETINGS_DROP = {"pandemonium_oracle", "dicow_v3_3"}  # the card's meeting table has no such rows
CARD_COLUMNS = {"icsi": "M1", "dipco": "M2"}
HEADERS = {"notsofar": "NSF", "ami": "AMI", "icsi": "ICSI", "mixer6": "Mx6", "sbc": "SBC", "dipco": "DiPCo",
           "nsf_ihm_mix": "NSF-IHM-mix", "ami_ihm_mix": "AMI-IHM-mix", "lsmix1": "LSMix 1", "lsmix2": "LSMix 2",
           "lsmix3": "LSMix 3", "libri2mix": "L2Mix", "libri2mix_noisy": "L2Mix noisy", "libri3mix": "L3Mix",
           "libri3mix_noisy": "L3Mix noisy"}
# (title, corpora, metrics shown per cell); Gemini is reported as cpWER on the meetings (its timestamps drift)
TABLES = [("Meetings, tcpWER (5 s collar)", bench.MEETINGS, ["tcpWER@5"]),
          ("Headset mixes, tcpWER / cpWER", bench.IHM_MIX, ["tcpWER@5", "cpWER"]),
          ("Short fully-overlapped mixtures, cpWER", bench.MIXTURES, ["cpWER"])]


def mean(values):
    """Mean of the displayed (2-decimal) cells, rounded half up, so a reader can recompute it from the table."""
    return (sum(Decimal(f"{v:.2f}") for v in values) / len(values)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def load(system):
    path = os.path.join(bench.system_dir(system), "scores.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def table(title, corpora, metrics, card):
    rows = []
    for label, system in SYSTEMS:
        if card and (system in CARD_DROP or (corpora == bench.MEETINGS and system in CARD_MEETINGS_DROP)):
            continue
        scores = load(system)
        if not any(c in scores for c in corpora):
            continue
        ms = ["cpWER"] if system == "gemini" and corpora == bench.MEETINGS else metrics
        cells = [" / ".join(f"{scores[c][m]:.2f}" for m in ms) if c in scores else "–" for c in corpora]
        complete = all(c in scores for c in corpora)
        avg = " / ".join(str(mean([scores[c][m] for c in corpora])) for m in ms) if complete else "–"
        rows.append((sum(scores[c][ms[0]] for c in corpora) if complete else float("inf"), label, cells, avg))
    heads = [CARD_COLUMNS.get(c, HEADERS[c]) if card else HEADERS[c] for c in corpora]
    lines = [f"### {title}", "", "| system | " + " | ".join(heads) + " | Avg |", "|---" * (len(corpora) + 2) + "|"]
    lines += [f"| {name} | " + " | ".join(cells) + f" | {avg} |" for _, name, cells, avg in sorted(rows)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", action="store_true")
    a = ap.parse_args()
    print("\n\n".join(table(*t, a.card) for t in TABLES))


if __name__ == "__main__":
    main()
