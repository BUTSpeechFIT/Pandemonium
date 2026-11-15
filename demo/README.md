# Pandemonium Demo

A lightweight [Gradio](https://www.gradio.app/) demo of **Pandemonium**, a multi-talker ASR model:
a diarization-conditioned [Parakeet-TDT-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
encoder (DiCoP) whose speaker streams talk to each other through cross-speaker attention and
re-estimate their own diarization mask. Given a recording of up to 30 minutes, it tells you **who said
what**.

## How it works

1. **Diarization** — [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
   (NeMo `SortformerEncLabelModel`, offline 30.4 s setting) produces per-speaker activity
   probabilities at 10 ms resolution for up to 8 speakers, in order of first appearance.
2. **STNO conditioning, soft** — the probabilities are averaged into Pandemonium's 80 ms frames and
   turned into one silence / target / non-target / overlap mask per speaker with the *same* formula
   used in training, evaluated on probabilities instead of 0/1 values. No RTTM, no threshold.
   (`DemoPipeline.transcribe(..., soft=False)` conditions on the thresholded segments instead; the
   UI always uses soft, which measured better, see below.)
3. **Transcription** — Pandemonium encodes all speakers of the recording in one pass (the mel
   front-end and subsampling run once and are shared), and a greedy CTC decode yields time-stamped
   words per speaker.

## Requirements

- Linux with an NVIDIA GPU. See [Memory](#memory) for what a long recording needs.
- [conda](https://docs.conda.io/) and `git`. Uploads are audio only: WAV, FLAC, MP3, M4A/AAC, OGG/Opus or WebM
  (video files are refused); non-WAV formats are decoded with a static `ffmpeg` that `imageio-ffmpeg` installs, so no system
  ffmpeg is needed.
- Internet access to the Hugging Face Hub on the first run, to download the models (no token needed; all of
  them are public).

Weights are downloaded on first launch: `BUT-FIT/Pandemonium` (2.8 GB),
`nvidia/Nemotron-3-Diarization` (0.4 GB), and Pandemonium's tokenizer, which is shared with
Parakeet-TDT-0.6b-v3 (taken from the model repo if present, otherwise extracted from
`nvidia/parakeet-tdt-0.6b-v3`, a one-time 2.5 GB download).

## Installation

From this `demo/` directory:

```bash
conda create -n pandemonium-demo python=3.12 -y
conda activate pandemonium-demo
pip install Cython packaging
pip install -r requirements.txt
```

NeMo is installed from a pinned `main` commit: Nemotron-3-Diarization needs a RoPE transformer
encoder that no PyPI release of NeMo has yet (`nemo-toolkit==3.0.0` refuses to load it).

The demo is self-contained: `pandemonium/` is an inference-only copy of the model code, so the rest
of this repository is not needed at runtime.

## Running

```bash
python app.py
```

Then open <http://127.0.0.1:7860>. Environment variables:

| variable | default | meaning |
|---|---|---|
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `127.0.0.1` / `7860` | where to serve |
| `GRADIO_ROOT_PATH` | — | when served behind a reverse proxy under a sub-path |
| `PANDEMONIUM_DTYPE` | `bf16` | `bf16` (as evaluated in the paper), `fp16` or `fp32` |
| `PANDEMONIUM_REPO` / `PANDEMONIUM_CKPT` | `BUT-FIT/Pandemonium` / `cp_wer=0.1597-epoch=16.ckpt` | which weights |
| `SPEAKER_THRESHOLD` | `0.5` | a diarizer channel counts as a speaker once it crosses this |
| `DEMO_SMTP_USER` / `DEMO_SMTP_PASSWORD_FILE` | — / `~/.config/pandemonium-demo/smtp_password` | FIT mail account that sends codes and notifications |
| `DEMO_ADMIN_EMAIL` | `ipoloka@fit.vut.cz` | gets signup notifications; shown to users for quota requests |
| `DEMO_DEFAULT_QUOTA` / `DEMO_IP_QUOTA` / `DEMO_IP_WINDOW_DAYS` | `5` / `10` / `30` | transcriptions per person / per IP per window |
| `MAX_SECONDS` | `2400` | longest accepted recording; the page advertises 30 minutes (`access.LIMIT`) and quietly allows up to 40 (checked from the file header before decoding) |
| `PANDEMONIUM_OFFLOAD` | `1` | keep both models in host RAM and move each to the GPU only while it runs; one GPU job at a time |
| `DEMO_EXAMPLE` | `ami_IS1009a` | which folder of `example/` the public example page shows |
| `DEMO_ACCESS_CODE` / `DEMO_ACCESS_CODE_FILE` | — | access code that gates the page and its API; **required** whenever the app is reachable beyond localhost |

## Access and quotas

Transcribing is always behind a login. Visitors who are not logged in land on **`/example/`**: a finished
transcript of an AMI meeting (`example/`, precomputed, so nothing runs and no quota is spent) with the
same timeline, player and views as the app, plus the corpus's human transcript and the scores. From
there, or from the login form (`/?login`), people request a personal code at **`/signup`** (linked from
the login page): name, email, affiliation, intended use, and agreement not to use the outputs to
train or distil other models. The code is **emailed** to them, which also proves they own the
address, and you get an email for every new signup. They log in with their email as the username
and the code as the password; your master code (`DEMO_ACCESS_CODE[_FILE]`) logs in as `admin`
with no limits.

- **Quota:** each *transcription* costs one request (finding speakers and playback are free);
  5 per person by default. Counted before the run starts and given back if it fails.
- **Against account farming:** one account per email, with aliases folded together (case,
  `+tags`, Gmail dots); disposable email domains are refused (`disposable_email_domains.txt`, an
  open blocklist); people on the default quota also share a per-IP cap (`DEMO_IP_QUOTA`
  transcriptions per `DEMO_IP_WINDOW_DAYS`, across all their accounts); the signup form accepts at
  most `DEMO_SIGNUPS_PER_IP_HOUR` submissions per IP per hour. Behind the nginx proxy the visitor's
  IP is the last public address in `X-Forwarded-For`, which a client cannot forge.
- **Codes:** asking again for an existing address emails a second code; the earlier one keeps
  working until the new one is first used, so nobody can lock a user out by re-requesting their
  address. The page shows the same message whatever the address's status, and at most
  `DEMO_CODES_PER_EMAIL_DAY` (3) code emails go to one address per day.
- **Privacy:** a public `/privacy` page (linked from the login and signup pages, and acknowledged at
  signup) says what is collected and why; `python admin.py delete EMAIL` erases a person's account and
  every record of them on request.
- **Storage:** one SQLite file (`DEMO_DB`, default `~/.config/pandemonium-demo/access.db`, mode 600).
  Codes are stored only as SHA-256 hashes.

Managing users, on the server:

```bash
python admin.py list                     # everyone, with usage
python admin.py grant someone@uni.edu 50 # set a quota
python admin.py add   someone@uni.edu 10 # give 10 more
python admin.py revoke someone@uni.edu   # their code stops working immediately
python admin.py create someone@uni.edu --name "Name" --quota 20   # add someone yourself
python admin.py show  someone@uni.edu    # details and every request
python admin.py ip    1.2.3.4            # what one address has been doing
python admin.py delete someone@uni.edu  # erase them entirely (GDPR request)
```

**Email** goes through FIT's server with your account (authenticated submission; the server
refuses unauthenticated mail). Put your password in a file only you can read and point the app
at your login:

```bash
install -m 600 /dev/null ~/.config/pandemonium-demo/smtp_password   # then write the password into it
export DEMO_SMTP_USER=<your FIT login>
```

Until that is set up, signups fail politely ("could not send the email") and create no account.

## Using the interface

1. Upload an **English** audio recording (up to 30 minutes) or record one from the microphone.
2. **Find speakers** draws a scrollable timeline: the waveform, then one lane per detected speaker with
   the diarizer's speech probability shaded in the background and its segments as blocks.
3. **Transcribe** (it diarizes first if needed) fills each lane with that speaker's turns, text
   included. Zoom with **−** / **+** / **Fit** or Ctrl/⌘ + wheel, scroll sideways, hover a block for
   its full text and times, and **click any block to play it**.
4. Below the timeline: the interleaved **Conversation**, a **Per speaker** view, and a searchable
   **Segments** table (click a row to play it). Download the result as
   [SegLST](https://github.com/fgnt/meeteval) JSON, per utterance or per word.

## Measured

NOTSOFAR-1 eval (160 sessions, 16.7 h), this exact pipeline, cpWER:

| conditioning | cpWER |
|---|---|
| oracle diarization (paper setup, reproduced by this code) | 14.14 |
| Nemotron-3-Diarization, **soft** probabilities (the demo) | **17.93** |
| Nemotron-3-Diarization, hard segments | 18.52 |
| the paper's real-diarization RTTMs | 22.41 |

The whole set took 117 s to diarize and ~3 min to transcribe on one GPU. A 60-minute, 8-speaker
recording peaks at ~12 GiB of GPU memory.

## Deployment (pcspeech-demo)

`deploy/` holds what runs on pcspeech-demo.fit.vut.cz: `run_server.sh` (port 7861, root path
`/pandemonium`), `pandemonium-demo.service` (systemd, logs in `~/.config/pandemonium-demo/service.{out,err}`),
`nginx-pandemonium.conf` (the `/pandemonium/` and `/dicow/` locations, to paste into the host's HTTPS
server block), `pandemonium-mail.py` (sends the emails from pcspeech4, which keeps the SMTP password),
and the static page now served at `/dicow/`.
Measured there (RTX A4000 16 GB, shared): 40 min x 8 speakers peaks at 10.2 GiB and transcribes in
22 s; between requests the demo holds no GPU memory.

## Example data and its licence

`example/ami_IS1009a/` (shown on the public example page) contains material from the **AMI Meeting
Corpus**, © University of Edinburgh and partners (Carletta et al., 2005), used under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/);
source: <https://groups.inf.ed.ac.uk/ami/corpus/>, meeting IS1009a, single distant microphone (Array1-01).
Changes made: the audio is re-encoded to mono 16 kHz MP3; `reference.seglst.json` is AMI's manual
transcript converted to SegLST (text unchanged); all other transcripts, the diarization and the scores
are produced by this demo. This licence covers only that data, not the rest of the repository. The
attribution is also shown on the example page itself.

## Limitations

- Trained on English meeting data; other languages and domains may be unreliable.
- At most 8 speakers (both the diarizer and Pandemonium's speaker slots).
- The model was trained on hard (0/1) masks; soft conditioning feeds it values in between, which
  it never saw in training. See the numbers below for what that does in practice.

## Contact

- [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
- [BUT Speech@FIT](https://github.com/BUTSpeechFIT), Brno University of Technology
