"""Multi-talker target-speaker ASR dataset.

Ported from `nemo/collections/asr/data/audio_to_text_and_stno.py`, keeping only the code
this model actually runs: the collate function, the manifest processor, and the BPE dataset.
NeMo's `_AudioTextDataset` / `AudioToBPEAndSTNODataset` split exists upstream only so the
character-level dataset can share the base; with that gone the two are merged here.

One dataset item is a **(session, target speaker)** pair:

  - `train`: one random target speaker per session per epoch, so an epoch sees every session
    once and the speaker choice is resampled each time.
  - `val=True`: every (session, speaker) pair exactly once, so validation is deterministic
    and covers all speakers. This is why `setup_validation_data` / `setup_test_data` pass
    `val=True`.

The transcript is the concatenation of the target speaker's segments; the STNO mask is built
from *all* speakers' segments so the model knows where the interfering speech is.

`multi_speaker=True` switches to a multi-talker layout instead (no model in this repository
uses it): one item is a whole session carrying one transcript per speaker
slot, ordered by who speaks first, and no mask is built at all — that model takes no
diarization input and is supervised by the transcripts alone.
"""

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType, VoidType
from nemo.utils import logging

from src.data.collections import ASRAudioTextSTNO
from src.data.stno import (
    SpeechSegment,
    create_stno_masks,
    frame_rate_from_downsampling_factor,
    num_encoder_frames,
    speaker_activity_mask,
)
from src.data.text_norm import get_text_norm

__all__ = [
    'AudioToBPEAndSTNODataset',
    'ASRManifestProcessor',
    'speech_collate_fn',
    'build_slot_activity',
    'build_slot_transcripts',
    'build_sot_transcript',
    'build_speaker_group',
    'group_collate_fn',
    'log_speaker_slot_capacity',
    'multi_speaker_collate_fn',
    'speakers_by_first_occurrence',
    'speakers_by_speaking_time',
    'slot_ordering',
    'SLOT_ORDERINGS',
]


def log_speaker_slot_capacity(speaker_counts, num_slots: int) -> None:
    """Report sessions with more speakers than there are slots to put them in.

    Their surplus speakers have no slot, so their words are absent from training and are charged
    as deletions at scoring — a floor on cpWER that no amount of training removes. Worth seeing
    at startup rather than inferring from a score that will not come down.
    """
    counts = list(speaker_counts)
    over = [count for count in counts if count > num_slots]
    if not over:
        return
    logging.warning(
        "%d of %d sessions have more than %d speakers (up to %d); the surplus speakers have no "
        "slot and will be scored as deletions. Raise `num_speaker_slots` (and the encoder's "
        "`global_tokens` with it) to cover them.",
        len(over),
        len(counts),
        num_slots,
        max(over),
    )


def speakers_by_first_occurrence(segments) -> List[str]:
    """Speaker labels ordered by when each of them first speaks.

    Slot 0 is whoever opens the session, slot 1 whoever speaks next, and so on. Ties break on the
    label, so two speakers starting in the same segment still get a deterministic order.

    Brittle as a *fixed* target, which is why `speakers_by_speaking_time` exists: "the sixth
    person to say anything" turns on five other onsets, so one early backchannel reshuffles every
    slot below it. On NOTSOFAR that showed as quality decaying with slot index -- slot 0
    near-verbatim, slot 3 at 47 words, slots 4+ silent. Under `slot_assignment: pit` it is no
    longer a target but a coordinate for `order_bias_weight`, which is a preference the search
    can overrule.
    """
    first = {}
    for segment in segments:
        speaker, start = segment['speaker'], segment['start']
        if speaker not in first or start < first[speaker]:
            first[speaker] = start
    return sorted(first, key=lambda speaker: (first[speaker], speaker))


def speakers_by_speaking_time(segments) -> List[str]:
    """Speaker labels ordered by how much they speak, the most talkative first.

    The more stable alternative to `speakers_by_first_occurrence`. Total speaking time is an
    aggregate over the whole session, so it does not hinge on a single onset the way a
    first-occurrence rank does: one stray early word cannot permute the slots. It also puts the
    speakers carrying the most words in the low slots, which are the ones that train most
    reliably, and leaves the sparse speakers -- the ones that are lost either way -- at the tail.

    Ties break on the label. Both orderings share the property the model needs: they are
    functions of the audio alone, so a model given no diarization can in principle reproduce
    them.
    """
    total = defaultdict(float)
    for segment in segments:
        total[segment['speaker']] += segment['duration']
    return sorted(total, key=lambda speaker: (-total[speaker], speaker))


# Slot orderings the dataset can be built with, by config name.
SLOT_ORDERINGS = {
    'first_occurrence': speakers_by_first_occurrence,
    'speaking_time': speakers_by_speaking_time,
}


def slot_ordering(name: str):
    """Resolve a `slot_order` config value to the function that implements it."""
    if name not in SLOT_ORDERINGS:
        raise ValueError(f"Unknown slot_order {name!r}; expected one of {sorted(SLOT_ORDERINGS)}.")
    return SLOT_ORDERINGS[name]


def build_slot_transcripts(segments, num_slots: int, pad_id: int, wrap, order_by='first_occurrence'):
    """One transcript per speaker slot, padded to a fixed `num_slots`.

    Slot `k` holds the `k`-th speaker under `order_by` (see `SLOT_ORDERINGS`), as the
    concatenation of their segments in time order. Slots past the session's speaker count
    are left empty, which is a zero-length CTC target: the loss for such a slot is the
    probability of the all-blank path, so the model is trained to emit nothing there rather than
    being left unsupervised.

    Args:
        segments: The session's tokenized segments, each a dict with `start`, `speaker` and a
            `text` list of token ids.
        num_slots: Fixed number of speaker slots the model has.
        pad_id: Token id to pad the ragged slots with. Never scored -- the per-slot lengths are
            what the CTC loss reads.
        wrap: Callable applied to one slot's token list, returning `(tokens, length)`. This is
            where BOS/EOS are added, and it is only applied to slots that have any tokens.
        order_by: Which slot ordering to assign speakers with, a key of `SLOT_ORDERINGS`.

    Returns:
        `(tokens, lengths, num_speakers)` -- a `(num_slots, U)` long tensor, a `(num_slots,)`
        long tensor, and the session's true speaker count *before* truncation to `num_slots`.
    """
    # Sorted by start regardless of the slot ordering: this decides the order of words *within*
    # a slot's transcript, which is always chronological. `order_by` decides which slot a
    # speaker lands in, which is a separate question.
    ordered = sorted(segments, key=lambda segment: segment['start'])
    speakers = slot_ordering(order_by)(ordered)

    slots = []
    for speaker in speakers[:num_slots]:
        tokens = [token for segment in ordered if segment['speaker'] == speaker for token in segment['text']]
        slots.append(wrap(tokens) if tokens else ([], 0))
    slots.extend(([], 0) for _ in range(num_slots - len(slots)))

    width = max((length for _, length in slots), default=0)
    padded = torch.full((num_slots, width), pad_id, dtype=torch.long)
    for index, (tokens, length) in enumerate(slots):
        if length:
            padded[index, :length] = torch.tensor(tokens, dtype=torch.long)

    lengths = torch.tensor([length for _, length in slots], dtype=torch.long)
    return padded, lengths, len(speakers)


def build_slot_activity(segments, num_slots: int, num_frames: int, frame_rate: float, order_by='first_occurrence'):
    """When each reference speaker is talking, `(num_slots, num_frames)` and binary.

    Column `k` holds the same speaker `build_slot_transcripts` puts in column `k`, so a slot
    matched to a reference can be supervised on both what it should say and when.

    The only frame-level signal the multi-talker model has: transcripts say what each speaker
    said, never when, which leaves the separator's frame gate to infer activity from CTC alone.
    On sequential audio it does not -- the 16614-step run put 76% of AMI's hypothesis words in
    slot 0 while transcribing the session to 31% WER. Rasterizing the diarization the manifests
    already carry is free at training time and unused at inference.

    Slots past the speaker count are all-zero, which is the right target.
    """
    ordered = sorted(segments, key=lambda segment: segment['start'])
    speakers = slot_ordering(order_by)(ordered)[:num_slots]
    mask = speaker_activity_mask(
        [
            SpeechSegment(start=segment['start'], duration=segment['duration'], speaker=segment['speaker'])
            for segment in ordered
        ],
        speakers,
        num_frames,
        frame_rate,
    )
    if mask.shape[0] < num_slots:
        mask = torch.nn.functional.pad(mask, (0, 0, 0, num_slots - mask.shape[0]))
    return mask


def build_utterance_spec(segments, num_slots: int, order_by='first_occurrence'):
    """The session's utterances kept separate, with their tokens, speaker and timing.

    The other three builders each flatten the session in some way -- into per-slot transcripts,
    into a per-slot activity mask, into one serialized sequence -- and all three throw away which
    utterance a token came from. The shuffle head needs exactly that, because its target is the
    set of token orders the timing permits, and the timing is per utterance.

    Speakers are numbered by `order_by`, the same numbering `build_slot_transcripts` uses for its
    slots, so a speaker index means the same thing across the heads. Utterances of speakers past
    `num_slots` are dropped, matching the slot truncation there.

    Returns a list of `(token_ids, speaker_index, start, duration)`, in start order. Plain Python
    rather than tensors: it is ragged in two dimensions and it crosses a dataloader worker
    boundary, which lists do and k2 graphs do not.
    """
    ordered = sorted(segments, key=lambda segment: segment['start'])
    speakers = slot_ordering(order_by)(ordered)[:num_slots]
    index = {speaker: position for position, speaker in enumerate(speakers)}

    return [
        (
            list(segment['text']),
            index[segment['speaker']],
            float(segment['start']),
            float(segment['duration']),
        )
        for segment in ordered
        if segment['speaker'] in index and len(segment['text']) > 0
    ]


def build_sot_transcript(segments, bos_id: int, eos_id: int, speaker_change_id: int, order_by='speaking_time'):
    """One serialized transcript for the whole session (Serialized Output Training).

    Every speaker's segments, in chronological order within the speaker, concatenated speaker by
    speaker under `order_by` (see `SLOT_ORDERINGS` -- the same two orderings `build_slot_transcripts`
    uses) and joined by `speaker_change_id`. Unlike `build_slot_transcripts` there is one sequence
    per session rather than one per fixed slot: an autoregressive decoder reads who's-talking off
    its own previous output (the speaker-change tokens it emitted), not off a slot index, so
    nothing here truncates or pads to a speaker count.

    Args:
        segments: The session's tokenized segments, each a dict with `start`, `speaker` and a
            `text` list of token ids.
        bos_id, eos_id: Token ids wrapping the whole sequence once, not per speaker.
        speaker_change_id: Token id inserted between (not around) two speakers' spans.
        order_by: Which speaker ordering to serialize in, a key of `SLOT_ORDERINGS`.

    Returns:
        `(tokens, length, num_speakers)` -- a `(U,)` long tensor, its length, and the session's
        speaker count.
    """
    ordered = sorted(segments, key=lambda segment: segment['start'])
    speakers = slot_ordering(order_by)(ordered)

    flat = [bos_id]
    for index, speaker in enumerate(speakers):
        if index:
            flat.append(speaker_change_id)
        flat.extend(token for segment in ordered if segment['speaker'] == speaker for token in segment['text'])
    flat.append(eos_id)

    return torch.tensor(flat, dtype=torch.long), len(flat), len(speakers)


def build_speaker_group(segments, num_slots, num_frames, frame_rate, wrap, pad_id, perturb=None):
    """Per-speaker STNO masks and transcripts for one whole session.

    The grouped ("cross-speaker") layout: one item is a session, carrying up to `num_slots` speakers,
    each with its own target-speaker STNO mask and transcript. Speaker `i` is the `i`-th speaker of
    the session by label (the same order `_process_sample` uses), and its mask is exactly what that
    path would build for that target speaker -- `create_stno_masks` distinguishes only the target
    row, so the *interfering* channels see every speaker, including any past the `num_slots` cap.
    Those capped speakers get no stream of their own and are scored as deletions, matching the
    multi-talker head's slot truncation.

    Args:
        segments: The session's tokenized segments, each a dict with `start`, `duration`,
            `speaker` and a `text` list of token ids.
        num_slots: Maximum speakers to emit a stream for.
        num_frames: Encoder frames the audio subsamples to; the mask width.
        frame_rate: Encoder frame rate, for rasterizing the activity mask.
        wrap: Callable adding BOS/EOS to one speaker's token list, returning `(tokens, length)`.
        pad_id: Token id the ragged per-speaker transcripts are padded with.
        perturb: Optional callable `(mask) -> mask` applied to each speaker's mask (training-time
            diarization augmentation); `None` leaves the reference mask.

    Returns:
        `(stno_masks, tokens, token_lengths, num_speakers)` -- a `(G, 4, num_frames)` float tensor,
        a `(G, U)` long tensor, a `(G,)` long tensor, and `G` (the emitted speaker count, i.e.
        `min(num_slots, session speaker count)`).
    """
    all_speakers = sorted({segment['speaker'] for segment in segments})
    kept = all_speakers[:num_slots]
    g = len(kept)

    spk_activity = speaker_activity_mask(
        [SpeechSegment(start=s['start'], duration=s['duration'], speaker=s['speaker']) for s in segments],
        all_speakers,
        num_frames,
        frame_rate,
    )

    masks, token_rows = [], []
    for target_idx in range(g):
        mask = create_stno_masks(spk_activity, target_idx)
        if perturb is not None:
            mask = perturb(mask)
        masks.append(mask)

        speaker = kept[target_idx]
        tokens = [token for segment in segments if segment['speaker'] == speaker for token in segment['text']]
        token_rows.append(wrap(tokens))

    width = max((length for _, length in token_rows), default=1)
    tokens = torch.full((g, width), pad_id, dtype=torch.long)
    for index, (toks, length) in enumerate(token_rows):
        tokens[index, :length] = torch.tensor(toks, dtype=torch.long)
    token_lengths = torch.tensor([length for _, length in token_rows], dtype=torch.long)

    return torch.stack(masks), tokens, token_lengths, g


def group_collate_fn(batch, pad_id: int):
    """Collate whole-session items that carry one STNO mask + transcript per speaker.

    Audio is stored once per session (the streams share it) and expanded to one row per speaker
    inside the model. Sessions differ in speaker count, so the speaker axis is padded to `Gmax`;
    padding slots repeat the session's first speaker's mask -- a valid forward that keeps batch-norm
    statistics clean -- and carry a zero-length transcript. `num_speakers` says which slots are
    real, and only those reach the loss and the metric.

    Args:
        batch: sequence of 7-tuples `(audio, audio_len, tokens (G,U), token_lens (G,),
            stno_masks (G,4,T), utt_id, num_speakers)`.
        pad_id: token id used to pad transcripts.

    Returns:
        `(audio (S,maxsamp), audio_len (S,), tokens (S,Gmax,maxU), token_lens (S,Gmax),
        stno_masks (S,Gmax,4,maxT), stno_mask_len (S,), utt_ids (S,), spk_ids (S,Gmax),
        num_speakers (S,))`.
    """
    audio, audio_lengths, tokens, token_lengths, stno_masks, utt_ids, num_speakers = zip(*batch)

    s = len(batch)
    g_max = max(item.shape[0] for item in stno_masks)
    max_samples = max(length.item() for length in audio_lengths)
    max_frames = max(item.shape[-1] for item in stno_masks)
    max_tokens = max(1, max(int(lengths.max()) if lengths.numel() else 0 for lengths in token_lengths))

    padded_audio = torch.stack([
        torch.nn.functional.pad(sig, (0, max_samples - length.item())) if length.item() < max_samples else sig
        for sig, length in zip(audio, audio_lengths)
    ])

    tokens_out = torch.full((s, g_max, max_tokens), pad_id, dtype=torch.long)
    token_len_out = torch.zeros((s, g_max), dtype=torch.long)
    masks_out = torch.zeros((s, g_max, 4, max_frames), dtype=stno_masks[0].dtype)
    spk_ids = torch.arange(g_max, dtype=torch.int32).unsqueeze(0).repeat(s, 1)

    for i, (toks, lens, mask) in enumerate(zip(tokens, token_lengths, stno_masks)):
        g = mask.shape[0]
        tokens_out[i, :g, : toks.shape[-1]] = toks
        token_len_out[i, :g] = lens
        masks_out[i, :g, :, : mask.shape[-1]] = mask
        # Padding speaker slots repeat speaker 0's mask, so their encoder forward is well-formed
        # (batch-norm sees plausible frames) even though they are excluded everywhere downstream.
        if g < g_max:
            masks_out[i, g:, :, : mask.shape[-1]] = mask[0]

    return (
        padded_audio,
        torch.stack(audio_lengths),
        tokens_out,
        token_len_out,
        masks_out,
        torch.tensor([item.shape[-1] for item in stno_masks], dtype=torch.long),
        torch.tensor(utt_ids, dtype=torch.int32),
        spk_ids,
        torch.tensor(num_speakers, dtype=torch.long),
    )


def multi_speaker_collate_fn(batch, pad_id: int):
    """Collate items that carry one transcript per speaker slot rather than one in total.

    Also serves the SOT dataset's single serialized-sequence items: padding here only ever
    touches the tensor's last dimension, so it is indifferent to whether `transcripts` carries a
    leading `num_slots` dimension or not.

    Args:
        batch: sequence of 8-tuples `(audio, audio_len, transcripts, transcript_lens, utt_id,
            num_speakers, slot_activity, utterances)`, where `transcripts` is `(num_slots, U)`,
            `transcript_lens` is `(num_slots,)` and `slot_activity` is `(num_slots, frames)` --
            or `(0, 0)` from the SOT dataset, which has no slots to supervise. Every item has the
            same number of slots, so only `U`, the frame count and the audio length vary across
            the batch.

    Returns:
        `(audio_signal, audio_lengths, transcripts, transcript_lengths, utt_ids, num_speakers,
        slot_activity, utterances)` with `transcripts` of shape `(B, num_slots, U)`, `transcript_lengths`
        `(B, num_slots)` and `slot_activity` `(B, num_slots, frames)`. A slot whose speaker is
        absent has length 0 and contributes an empty CTC target and an all-zero activity target.
        `utterances` stays a per-item list -- it is ragged in two dimensions and only the
        shuffle head reads it, which builds a graph per item anyway.
    """
    (audio, audio_lengths, transcripts, transcript_lengths, utt_ids, num_speakers, activity,
     utterances) = zip(*batch)

    max_audio_len = max(audio_lengths).item()
    max_frames = max(item.shape[-1] for item in activity)
    max_tokens_len = max(int(lengths.max()) for lengths in transcript_lengths)
    # A batch in which every slot is empty would otherwise collate to a zero-width tensor, which
    # the CTC loss cannot index. One column of padding costs nothing and keeps the shape valid.
    max_tokens_len = max(max_tokens_len, 1)

    padded_audio, padded_transcripts = [], []
    for signal, signal_len, tokens in zip(audio, audio_lengths, transcripts):
        signal_len = signal_len.item()
        if signal_len < max_audio_len:
            signal = torch.nn.functional.pad(signal, (0, max_audio_len - signal_len))
        padded_audio.append(signal)

        if tokens.shape[-1] < max_tokens_len:
            tokens = torch.nn.functional.pad(tokens, (0, max_tokens_len - tokens.shape[-1]), value=pad_id)
        padded_transcripts.append(tokens)

    return (
        torch.stack(padded_audio),
        torch.stack(audio_lengths),
        torch.stack(padded_transcripts),
        torch.stack(transcript_lengths),
        torch.tensor(utt_ids, dtype=torch.int32),
        torch.stack(num_speakers),
        torch.stack(
            [
                torch.nn.functional.pad(item, (0, max_frames - item.shape[-1]))
                if item.shape[-1] < max_frames
                else item
                for item in activity
            ]
        ),
        list(utterances),
    )


def speech_collate_fn(batch, pad_id: int):
    """Collate a batch of (audio, tokens, STNO mask) tuples, zero-padding to the batch maximum.

    Args:
        batch: sequence of 8-tuples as returned by `AudioToBPEAndSTNODataset.__getitem__`.
        pad_id: token id used to pad transcripts.

    Returns:
        `(audio_signal, audio_lengths, tokens, tokens_lengths, stno_masks, stno_mask_lengths,
        utt_ids, spk_ids)`, where `utt_ids` indexes the manifest row and `spk_ids` indexes the
        target speaker within that session's sorted speaker list.
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) not in (8, 9):
        raise ValueError(f"Expected 8 or 9 tensors in the batch, got {len(packed_batch)}")

    # A 9th element is the session's speaker count, which only the Lhotse dataset supplies.
    num_speakers = packed_batch[8] if len(packed_batch) == 9 else None
    _, audio_lengths, _, tokens_lengths, _, stno_mask_lengths, utt_ids, spk_ids = packed_batch[:8]

    max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()
    max_stno_mask_len = max(stno_mask_lengths).item()

    audio_signal, tokens, stno_masks = [], [], []
    for item in batch:
        sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len = item[:6]
        sig_len = sig_len.item()
        if sig_len < max_audio_len:
            sig = torch.nn.functional.pad(sig, (0, max_audio_len - sig_len))
        audio_signal.append(sig)

        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            tokens_i = torch.nn.functional.pad(tokens_i, (0, max_tokens_len - tokens_i_len), value=pad_id)
        tokens.append(tokens_i)

        stno_mask_i_len = stno_mask_i_len.item()
        if stno_mask_i_len < max_stno_mask_len:
            stno_mask_i = torch.nn.functional.pad(stno_mask_i, (0, max_stno_mask_len - stno_mask_i_len))
        stno_masks.append(stno_mask_i)

    collated = (
        torch.stack(audio_signal),
        torch.stack(audio_lengths),
        torch.stack(tokens),
        torch.stack(tokens_lengths),
        torch.stack(stno_masks),
        torch.stack(stno_mask_lengths),
        torch.tensor(utt_ids, dtype=torch.int32),
        torch.tensor(spk_ids, dtype=torch.int32),
    )
    if num_speakers is not None:
        collated = collated + (torch.stack(num_speakers),)
    return collated


class ASRManifestProcessor:
    """Loads a multi-talker manifest and drops sessions with no usable transcript."""

    def __init__(
        self,
        manifest_filepath: Union[str, List[str]],
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.parser = parser

        self.collection = ASRAudioTextSTNO(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            parse_func=manifest_parse_func,
        )

        # A session whose every segment tokenized to nothing cannot supply a target transcript.
        ids_to_pop = [i for i in range(len(self.collection)) if not self.collection[i].text_tokens]
        if ids_to_pop:
            logging.info(
                "Dropping %d/%d sessions with no tokenizable transcript", len(ids_to_pop), len(self.collection)
            )
            for i in reversed(ids_to_pop):
                self.collection.pop(i)
        logging.info("Manifest loaded with %d sessions", len(self.collection))

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        return self.process_text_by_sample(self.collection[index])

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        return self.process_text_by_sample(self.collection[manifest_idx])

    def process_text_by_sample(self, text_tokens: List[int]) -> Tuple[List[int], int]:
        t, tl = text_tokens, len(text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


class _TokenizerWrapper:
    """Applies text normalization before tokenizing, so refs and hyps are normalized alike."""

    def __init__(self, tokenizer, text_norm_type: Optional[str] = 'whisper_nsf'):
        self.is_aggregate = isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer)
        self._tokenizer = tokenizer
        self.text_norm = get_text_norm(text_norm_type) if text_norm_type else (lambda x: x)

    def __call__(self, *args):
        if isinstance(args[0], List) and self.is_aggregate:
            t = []
            for span in args[0]:
                t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
            return t

        args = tuple(self.text_norm(x) for x in args)
        return self._tokenizer.text_to_ids(*args)


class AudioToBPEAndSTNODataset(Dataset):
    """Yields `(audio, tokens, stno_mask)` for one (session, target speaker) pair.

    Args:
        manifest_filepath: Path(s) to the JSON Lines manifest; comma-separated string allowed.
        tokenizer: A `nemo.collections.common.tokenizers.TokenizerSpec` subclass.
        sample_rate: Sample rate to resample loaded audio to.
        int_values: If True, load samples as 32-bit integers.
        augmentor: Optional `AudioAugmentor` applied to the loaded waveform.
        max_duration: Drop sessions longer than this.
        min_duration: Drop sessions shorter than this.
        max_utts: Limit the number of sessions.
        trim: Whether to trim leading/trailing silence.
        use_start_end_token: Add [BOS]/[EOS] around the target transcript.
        channel_selector: Select or average channels of multi-channel audio.
        manifest_parse_func: Optional custom manifest line parser.
        audio_downsampling_factor: Samples per encoder frame; sets the STNO mask frame rate.
        text_norm_type: Text normalization applied before tokenization (`None` disables it).
        val: If True, enumerate every (session, speaker) pair instead of sampling one speaker.
        multi_speaker: If True, one item is a whole session with one transcript per speaker
            slot, and no mask is built at all — the multi-talker model takes no diarization
            input. `val` is then irrelevant, since a session is emitted once either way.
        num_speaker_slots: Speaker slots per item in `multi_speaker` mode.
        slot_order: How speakers are assigned to slots; see `SLOT_ORDERINGS`.
        sot: If True, one item is a whole session with one *serialized* transcript covering
            every speaker (Serialized Output Training), and no mask is built — see
            `build_sot_transcript`. Mutually exclusive with `multi_speaker`.
        sot_order: Which speaker ordering to serialize in, under `sot`; see `SLOT_ORDERINGS`.
        sot_bos_id, sot_eos_id, sot_speaker_change_id: Token ids the serialized sequence is built
            with, under `sot`. Not read from the tokenizer like `use_start_end_token`'s BOS/EOS,
            since Parakeet's tokenizer has none trained in; these must be given explicitly.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'stno_masks': NeuralType(('B', 'S', 'T'), MaskType()),
            'stno_mask_length': NeuralType(tuple('B'), LengthsType()),
            'utterance_id': NeuralType(tuple('B'), VoidType()),
            'speaker_id': NeuralType(tuple('B'), VoidType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, List[str]],
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        channel_selector: Optional[ChannelSelectorType] = None,
        manifest_parse_func: Optional[Callable] = None,
        audio_downsampling_factor: int = 1,
        text_norm_type: Optional[str] = 'whisper_nsf',
        val: bool = False,
        multi_speaker: bool = False,
        num_speaker_slots: int = 8,
        slot_order: str = 'first_occurrence',
        sot: bool = False,
        sot_order: str = 'speaking_time',
        sot_bos_id: Optional[int] = None,
        sot_eos_id: Optional[int] = None,
        sot_speaker_change_id: Optional[int] = None,
        group_speakers: bool = False,
    ):
        if sot and (sot_bos_id is None or sot_eos_id is None or sot_speaker_change_id is None):
            raise ValueError("sot=True requires sot_bos_id, sot_eos_id and sot_speaker_change_id.")
        if group_speakers and (multi_speaker or sot):
            raise ValueError("group_speakers is mutually exclusive with multi_speaker and sot.")

        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=_TokenizerWrapper(tokenizer, text_norm_type),
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            manifest_parse_func=manifest_parse_func,
        )

        self.val = val
        self.multi_speaker = multi_speaker
        self.num_speaker_slots = num_speaker_slots
        self.slot_order = slot_order
        self.sot = sot
        self.sot_order = sot_order
        self.sot_bos_id = sot_bos_id
        self.sot_eos_id = sot_eos_id
        self.sot_speaker_change_id = sot_speaker_change_id
        self.group_speakers = group_speakers
        if val and not multi_speaker and not sot and not group_speakers:
            # Deterministic, exhaustive: one item per (session, speaker).
            self.per_spk_collection = [
                (sample, speaker)
                for sample in self.manifest_processor.collection
                for speaker in sorted({segment['speaker'] for segment in sample.text_tokens})
            ]
        if multi_speaker or group_speakers:
            log_speaker_slot_capacity(
                (len({segment['speaker'] for segment in sample.text_tokens})
                 for sample in self.manifest_processor.collection),
                num_speaker_slots,
            )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.channel_selector = channel_selector
        self.audio_downsampling_factor = audio_downsampling_factor
        self.frame_rate = frame_rate_from_downsampling_factor(sample_rate, audio_downsampling_factor)

    def __len__(self):
        if self.val and not self.multi_speaker and not self.sot and not self.group_speakers:
            return len(self.per_spk_collection)
        return len(self.manifest_processor.collection)

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        if self.multi_speaker:
            return self._process_multi_speaker_sample(index)
        if self.sot:
            return self._process_sot_sample(index)
        if self.group_speakers:
            return self._process_group_sample(index)
        return self._process_sample(index)

    def _process_group_sample(self, index):
        """One whole session, one STNO mask + transcript per speaker. See `build_speaker_group`."""
        sample = self.manifest_processor.collection[index]
        f, fl = self._load_audio(sample)
        stno_masks, tokens, token_lengths, num_speakers = build_speaker_group(
            sample.text_tokens,
            self.num_speaker_slots,
            num_encoder_frames(int(fl), self.audio_downsampling_factor),
            self.frame_rate,
            self.manifest_processor.process_text_by_sample,
            self.manifest_processor.pad_id,
        )
        return f, fl, tokens, token_lengths, stno_masks, sample.id, num_speakers

    def _load_audio(self, sample):
        features = self.featurizer.process(
            sample.audio_file,
            offset=sample.offset if sample.offset is not None else 0,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        return features, torch.tensor(features.shape[0]).long()

    def _process_multi_speaker_sample(self, index):
        """One session, one transcript per speaker slot. See `build_slot_transcripts`."""
        sample = self.manifest_processor.collection[index]
        f, fl = self._load_audio(sample)
        tokens, lengths, num_speakers = build_slot_transcripts(
            sample.text_tokens,
            self.num_speaker_slots,
            self.manifest_processor.pad_id,
            self.manifest_processor.process_text_by_sample,
            order_by=self.slot_order,
        )
        activity = build_slot_activity(
            sample.text_tokens,
            self.num_speaker_slots,
            num_encoder_frames(int(fl), self.audio_downsampling_factor),
            self.frame_rate,
            order_by=self.slot_order,
        )
        utterances = build_utterance_spec(
            sample.text_tokens, self.num_speaker_slots, order_by=self.slot_order
        )
        return (
            f, fl, tokens, lengths, sample.id, torch.tensor(num_speakers).long(), activity,
            utterances,
        )

    def _process_sot_sample(self, index):
        """One session, one serialized transcript. See `build_sot_transcript`."""
        sample = self.manifest_processor.collection[index]
        f, fl = self._load_audio(sample)
        tokens, length, num_speakers = build_sot_transcript(
            sample.text_tokens,
            self.sot_bos_id,
            self.sot_eos_id,
            self.sot_speaker_change_id,
            order_by=self.sot_order,
        )
        # No slots, so nothing to supervise: the collate carries an empty tensor through.
        return (
            f, fl, tokens, torch.tensor(length).long(), sample.id,
            torch.tensor(num_speakers).long(), torch.zeros(0, 0), [],
        )

    def _process_sample(self, index):
        if self.val:
            sample, target_speaker = self.per_spk_collection[index]
        else:
            sample = self.manifest_processor.collection[index]
            target_speaker = None

        f, fl = self._load_audio(sample)

        speakers = sorted({segment['speaker'] for segment in sample.text_tokens})
        if target_speaker is None:
            target_speaker = random.choice(speakers)
        target_idx = speakers.index(target_speaker)

        num_frames = num_encoder_frames(int(fl), self.audio_downsampling_factor)
        segments = [
            SpeechSegment(start=segment['start'], duration=segment['duration'], speaker=segment['speaker'])
            for segment in sample.text_tokens
        ]
        spk_activity_mask = speaker_activity_mask(segments, speakers, num_frames, self.frame_rate)
        stno_mask = create_stno_masks(spk_activity_mask, target_idx)

        # The transcript is every segment of the target speaker, in manifest order.
        target_tokens = []
        for segment in sample.text_tokens:
            if segment['speaker'] == target_speaker:
                target_tokens.extend(segment['text'])

        t, tl = self.manifest_processor.process_text_by_sample(target_tokens)

        return (
            f,
            fl,
            torch.tensor(t).long(),
            torch.tensor(tl).long(),
            stno_mask,
            torch.tensor(stno_mask.shape[-1]).long(),
            sample.id,
            target_idx,
        )

    def _collate_fn(self, batch):
        pad_id = self.manifest_processor.pad_id
        if self.multi_speaker or self.sot:
            return multi_speaker_collate_fn(batch, pad_id=pad_id)
        if self.group_speakers:
            return group_collate_fn(batch, pad_id=pad_id)
        return speech_collate_fn(batch, pad_id=pad_id)
