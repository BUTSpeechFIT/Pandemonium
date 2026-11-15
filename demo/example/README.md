# Precomputed examples

Each folder is one recording with the demo pipeline's complete output, so the UI can show a
finished result instantly, with no model running. Load one with `example.load_example(name)`,
which returns the same objects the app's renderers already take (`timeline`,
`render_conversation`, `render_per_speaker`).

## `ami_IS1009a`

A 14-minute design-team meeting with 4 speakers from the **AMI Meeting Corpus** test set,
recorded on a single distant microphone. Produced by the same pipeline as the live demo:
Nemotron-3-Diarization (soft probabilities) followed by Pandemonium.

Against AMI's human transcript (normalized as in the paper): **cpWER 14.73%**, tcpWER 15.44%
(5 s collar), 1,962 reference words. Each detected speaker maps to exactly one participant.

**Attribution (required by the licence):** AMI Meeting Corpus, University of Edinburgh and
partners (Carletta et al., 2005), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
<https://groups.inf.ed.ac.uk/ami/corpus/>. The audio is re-encoded to mono MP3; the transcripts
in `transcript*.json` are produced by Pandemonium, and `reference.seglst.json` is AMI's own.
Show this attribution wherever the example is shown.

## Files in an example folder

| file | contents |
|---|---|
| `meta.json` | title, description, duration, speaker count, source and licence, how it was produced, scores |
| `audio.mp3` | the recording, mono 96 kbps, for the browser player |
| `diarization.npz` | `probs` (speakers × frames, float16, the diarizer's soft speaker probabilities), `frame_shift` (s per frame, 0.01), `segments` (N × 3: start s, end s, speaker row) |
| `words.json` | per speaker, `[[word, start_s, end_s], ...]`, in speaker-row order |
| `utterances.json` | turns as the app shows them: `{speaker, start, end, text}`, sorted by start |
| `transcript.seglst.json` | the turns as [SegLST](https://github.com/fgnt/meeteval) (`Speaker 1`...), the app's "utterances" download |
| `transcript.words.seglst.json` | one SegLST segment per word, the app's "word timestamps" download |
| `reference.seglst.json` | *(optional)* the human reference, with the real speaker IDs and original text |
| `scores.json` | *(optional)* cpWER / tcpWER against the reference, error breakdown, speaker mapping |

Speaker rows are in order of first appearance, the same as in a live run, so `Speaker 1` in
`words.json`, the timeline and the transcripts is always the same person.

## Using it in the app

```python
from example import EXAMPLES_DIR, load_example

ex = load_example("ami_IS1009a")
plot = timeline(ex.audio_path, ex.audio, ex.diar, ex.utts)
conversation = render_conversation(ex.utts)
per_speaker = render_per_speaker(ex.utts, ex.diar.num_speakers)
```

The timeline's player streams `ex.audio_path` through Gradio's file route, which only serves
allowed folders, so add `allowed_paths=[EXAMPLES_DIR]` to `demo.launch(...)`. Showing an example
should not count against anyone's transcription quota: nothing runs.

## Adding another

```bash
python make_example.py meeting.wav my_meeting --title "My meeting (3 speakers, 10 min)" \
    --reference my_meeting_ref.seglst.json --source-json my_meeting_source.json
```

`--reference` scores the result (needs `meeteval`; the score is on raw text, without the paper's
normalization). `--source-json` is merged into `meta.json` as `source`: put the corpus, licence
and attribution there. Only add recordings you are allowed to redistribute.
