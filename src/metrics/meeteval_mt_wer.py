"""Multi-talker cpWER / tcpWER via meeteval, with STM dumping.

Ported from `nemo/collections/asr/metrics/meeteval_mt_wer.py` on the NeMo `ts_asr` branch,
with these changes:

  - `get_world_size` is defined locally; the NeMo function it imported does not exist in
    `nemo_toolkit==2.7.3`.
  - STM segments carry the real session id and the real speaker label instead of the
    manifest row index and the speaker's position in the sorted speaker list. cpWER and
    tcpWER are both speaker-permutation invariant, so this cannot change any reported
    number - it only makes `ref.stm` / `hyp.stm` readable and joinable with other tooling.
  - A duplicated (session, speaker) pair is marked processed *before* the empty-hypothesis
    early exits, so DDP's batch padding cannot double-count empty predictions.
  - A gt/prediction session count mismatch warns and lists the missing sessions instead of
    asserting, which previously aborted a whole evaluation.
  - `tcp_collar`, `output_per_word_timestamps`, `session_scope` and the text normalizer are
    configurable.
  - Scoring is sharded by session rather than by manifest row. cpWER optimizes a permutation
    per session, so splitting one across ranks made the result depend on the GPU count.

Ranks score disjoint session subsets and the insertion/deletion/substitution/length counts are
summed, which is identical to scoring everything on one rank but much faster, because cpWER's
speaker-permutation search is the expensive part.

Requires `compute_timestamps=True` on the decoding config: word timestamps are what place
the hypotheses on the time axis for tcpWER and for the STM.
"""

from math import ceil
from typing import Dict, List, Optional, Union

import meeteval
import torch
from meeteval.io.seglst import SegLST, SegLstSegment
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging
from nemo.utils.get_rank import get_rank, is_global_rank_zero

from src.data.text_norm import get_text_norm
from src.model.ctc_timestamps import emission_word_offsets

__all__ = ['MeetevalMTWER', 'pool_results']

# Hypotheses whose words are farther apart than this are split into separate STM segments
# when `output_per_word_timestamps` is False.
SEGMENT_SPLIT_GAP_SECONDS = 0.5


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def pool_results(results: List[Dict]) -> Dict:
    """Sum several cpWER/tcpWER results into one, as scoring their union in a single pass would.

    Every count is per (session, speaker), so the error and reference-length counts simply add up
    and the rate is recomputed from the totals. Used to reduce the per-rank shards of one dataset,
    and by the model to pool separately-scored datasets into the single number checkpoints are
    selected on.
    """
    pooled = {key: sum(res[key] for res in results) for key in ('ins', 'del', 'sub', 'len')}
    pooled['wer'] = (
        (pooled['sub'] + pooled['ins'] + pooled['del']) / pooled['len'] if pooled['len'] else 0.0
    )
    return pooled


class MeetevalMTWER(Metric):
    """Accumulates per-(session, speaker) hypotheses and scores them with cpWER / tcpWER.

    Args:
        decoding: The model's decoding object; also supplies the tokenizer.
        embed_duration: Seconds per encoder frame (0.08 for 12.5 Hz), used to turn decoded
            frame offsets into times.
        output_per_word_timestamps: Emit one STM segment per word (the default, and what makes
            tcpWER timings tightest) rather than grouping words into segments on pauses.
        tcp_collar: tcpWER collar in seconds.
        compute_tcpwer: Whether to compute tcpWER at all. tcpWER's time-constrained alignment is
            the expensive half of scoring -- it sorts and aligns every hypothesis on the time
            axis, and on long or untrained (runaway) hypotheses that dominates the eval wall
            clock. When False, `compute()` returns `None` for the tcpWER result and no `tcp_*`
            metric is logged; checkpoint selection uses `val/cp_wer`, which is unaffected. Leave
            True for heads whose timestamps are real and `val/tcp_wer` is meaningful.
        text_norm_type: Normalizer applied to both references and hypotheses before scoring.
        speaker_labels: How a hypothesis's speaker is named in the STM. `'reference'` looks the
            id up in the session's sorted speaker list, which is right when the id *is* a
            position in that list, as it is for the target-speaker model. `'slot'` names it
            `spk<id>` instead, for the multi-talker model, whose ids are output slots that
            correspond to no particular reference speaker. Neither choice can move a score:
            cpWER and tcpWER both search over speaker permutations, so the labels only decide
            how `hyp.stm` reads.
        session_scope: The unit one speaker permutation is solved over -- `'recording'` composes
            every manifest row of a recording onto one timeline, `'window'` scores each row
            independently. See `_session_key`; this one *does* move the score.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding],
        use_cer=False,
        log_prediction=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        fold_consecutive=True,
        sync_on_compute=True,
        embed_duration=0.08,  # 80 ms - 12.5 Hz with 8x downsampling conformer
        output_per_word_timestamps=True,
        tcp_collar=5,
        compute_tcpwer=True,
        text_norm_type='whisper_nsf',
        speaker_labels='reference',
        session_scope='recording',
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        if speaker_labels not in ('reference', 'slot'):
            raise ValueError(f"speaker_labels must be 'reference' or 'slot', got {speaker_labels!r}")
        self.speaker_labels = speaker_labels

        if session_scope not in ('recording', 'window'):
            raise ValueError(f"session_scope must be 'recording' or 'window', got {session_scope!r}")
        self.session_scope = session_scope

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index
        self.embed_duration = embed_duration
        self.output_per_word_timestamps = output_per_word_timestamps
        self.tcp_collar = tcp_collar
        self.compute_tcpwer = compute_tcpwer

        self.text_norm = get_text_norm(text_norm_type) if text_norm_type else (lambda x: x)

        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions,
                encoded_lengths=predictions_lengths,
                return_hypotheses=True,
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            # NeMo's CTC word offsets start at the previous word's emission; see ctc_timestamps.
            self.decode = lambda predictions, predictions_lengths: emission_word_offsets(
                self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=predictions,
                    decoder_lengths=predictions_lengths,
                    fold_consecutive=self.fold_consecutive,
                    return_hypotheses=True,
                )
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("preds_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("utt_ids", default=[], dist_reduce_fx="cat")
        self.add_state("spk_ids", default=[], dist_reduce_fx="cat")

    def _hypothesis_to_states(self, hypothesis, device):
        """Re-tokenize a hypothesis and extract its word timestamps as flat tensors.

        Re-tokenizing the decoded *text* (rather than keeping `y_sequence`) is what collapses
        the RNNT output into the token sequence the text actually corresponds to.
        """
        token_ids = torch.tensor(
            self.decoding.tokenizer.text_to_ids(hypothesis.text), dtype=torch.int32, device=device
        )

        if not hypothesis.text:
            return (
                token_ids,
                torch.tensor(len(token_ids), dtype=torch.int32, device=device),
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )

        words = hypothesis.text.split()
        word_timestamps = hypothesis.timestamp['word']
        if len(words) != len(word_timestamps):
            # NeMo occasionally returns a different number of word timestamps than the text has
            # words -- rarely, and depending on model state. Dropping the utterance over it costs
            # a whole session-speaker as a deletion, so align instead.
            #
            # `get_words_offsets` emits one entry per group of sub-word tokens and never checks
            # that the group decodes to exactly one whitespace-free word, so a surplus entry is
            # most likely one that decodes to nothing. Those are dropped first, which keeps the
            # rest aligned; only if that is not enough does this fall back to truncating, which
            # is right when the surplus is at the end and approximate otherwise.
            logging.warning(
                "MeetevalMTWER: %d words but %d word timestamps; realigning.",
                len(words),
                len(word_timestamps),
            )
            if len(word_timestamps) > len(words):
                word_timestamps = [w for w in word_timestamps if str(w.get('word', ' ')).strip()]
            word_timestamps = list(word_timestamps[: len(words)])
            while len(word_timestamps) < len(words):
                word_timestamps.append(word_timestamps[-1] if word_timestamps else {'start_offset': 0, 'end_offset': 0})

        return (
            token_ids,
            torch.tensor(len(token_ids), dtype=torch.int32, device=device),
            torch.tensor(
                [[w['start_offset'], w['end_offset']] for w in word_timestamps],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(len(word_timestamps), dtype=torch.int32, device=device),
        )

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        utt_ids: torch.Tensor,
        spk_ids: torch.Tensor,
    ):
        with torch.no_grad():
            device = predictions.device
            empty = (
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )

            try:
                decoded = self.decode(predictions, predictions_lengths)
                states = [self._hypothesis_to_states(hyp, device) for hyp in decoded]
            except Exception as exc:
                # NeMo occasionally returns a hypothesis whose word timestamps do not line up
                # with its text. Retry one utterance at a time and treat the broken ones as
                # empty hypotheses so the rest of the batch still scores.
                logging.error("MeetevalMTWER: batched decoding failed: %s", exc)
                logging.warning("MeetevalMTWER: falling back to one-by-one decoding.")
                states = []
                for i in range(len(predictions)):
                    try:
                        decoded = self.decode(predictions[i : i + 1], predictions_lengths[i : i + 1])
                        states.append(self._hypothesis_to_states(decoded[0], device))
                    except Exception as inner_exc:
                        logging.error("MeetevalMTWER: skipping utterance %d: %s", i, inner_exc)
                        states.append(empty)

            self._record_states(states, utt_ids, spk_ids, device)

    def update_from_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        utt_ids: torch.Tensor,
        spk_ids: torch.Tensor,
    ):
        """`update()` for a caller that already has decoded `Hypothesis` objects.

        For a model whose decoding is not CTC/RNNT log-probabilities -- SOT's autoregressive
        greedy decode, for instance -- `self.decode` has nothing to run: there is no `predictions`
        tensor to re-decode, only text already produced. This is `update()` from the point where
        `decoded` would have been assigned, so it goes through the same `_hypothesis_to_states`
        re-tokenization, and is scored identically.

        Each `hypothesis.timestamp['word']` still has to carry one `{'start_offset',
        'end_offset'}` entry per word: `tcpwer` reads those values, so a caller with no real
        per-word alignment (again, SOT) should not trust its `val/tcp_*` numbers even though this
        method runs -- only fill in placeholders so the STM/segment plumbing has something to
        read, and rely on `val/cp_*` instead, which never looks at them.
        """
        with torch.no_grad():
            device = utt_ids.device
            states = [self._hypothesis_to_states(hyp, device) for hyp in hypotheses]
            self._record_states(states, utt_ids, spk_ids, device)

    def _record_states(self, states, utt_ids: torch.Tensor, spk_ids: torch.Tensor, device):
        # Every utterance must be recorded, faulty ones included, or meeteval will later
        # complain about missing hypotheses.
        if not len(states) == len(utt_ids) == len(spk_ids):
            raise ValueError(
                f"Decoded {len(states)} hypotheses for {len(utt_ids)} utterance ids and "
                f"{len(spk_ids)} speaker ids."
            )

        for token_ids, num_tokens, word_timestamps, num_word_timestamps in states:
            self.preds.append(token_ids)
            self.preds_lengths.append(num_tokens)
            self.preds_word_timestamps.append(word_timestamps)
            self.preds_word_timestamps_lengths.append(num_word_timestamps)
        self.utt_ids.extend(utt_ids.detach())
        self.spk_ids.extend(spk_ids.detach())

    @staticmethod
    def _process_metric_res(res):
        output = {'wer': 0, 'ins': 0, 'del': 0, 'sub': 0, 'len': 0}
        for i in res:
            output['len'] += res[i].length
            output['ins'] += res[i].insertions
            output['del'] += res[i].deletions
            output['sub'] += res[i].substitutions
        return output

    def _session_key(self, sample) -> str:
        """The STM session id: the unit cpWER solves one speaker permutation over.

        `recording` composes every row of a recording onto one timeline, right when speaker
        labels mean the same thing across rows -- a target-speaker model given a diarization
        mask, or a manifest whose row *is* a session.

        `window` makes each row its own session, the honest unit for a model emitting anonymous
        per-window slots with no way to carry a slot's identity between windows. An oracle with
        perfect words and perfect per-window separation still scores 0.744 (NOTSOFAR 30 s dev)
        and 0.858 (AMI) under `recording`, against 0.0 under `window` -- so a recording-level
        number mostly measures cross-window linking, which is a separate capability from the
        separation and transcription being trained here.
        """
        if self.session_scope == 'window':
            return f"{sample.session_id}#{sample.id}"
        return sample.session_id

    def _reference_segments(self, targets_collection) -> Dict[int, List[SegLstSegment]]:
        """Build reference STM segments from the whole dataset, keyed by manifest row id.

        Segment `start` values are relative to the row's `offset`, so the offset is added to
        put every row on its recording's absolute timeline. That matters for segmented
        manifests, where many rows share one recording: without it their windows would all
        start at zero and pile up on top of each other. For full-session manifests the offset
        is zero and this is a no-op.
        """
        gt_segments = {}
        for sample in targets_collection:
            offset = sample.offset or 0.0
            for segment in sample.text_tokens:
                start = offset + segment['start']
                gt_segments.setdefault(sample.id, []).append(
                    SegLstSegment(
                        session_id=self._session_key(sample),
                        speaker=segment['speaker'],
                        words=self.text_norm(self.decoding.decode_tokens_to_str(segment['text'])),
                        start_time=start,
                        end_time=start + segment['duration'],
                    )
                )
        return gt_segments

    def _words_to_segments(
        self, words, word_timestamps, session_id, speaker, offset: float = 0.0
    ) -> List[SegLstSegment]:
        """Turn one hypothesis's words into STM segments, per word or grouped on pauses.

        `offset` shifts the window-relative decoded times onto the recording's absolute
        timeline, matching what `_reference_segments` does for the references.
        """
        if self.output_per_word_timestamps:
            return [
                SegLstSegment(
                    session_id=session_id,
                    speaker=speaker,
                    words=self.text_norm(word),
                    start_time=offset + start * self.embed_duration,
                    end_time=offset + end * self.embed_duration,
                )
                for word, (start, end) in zip(words, word_timestamps)
            ]

        segments = []
        group = [(words[0], *word_timestamps[0])]

        def flush(group):
            return SegLstSegment(
                session_id=session_id,
                speaker=speaker,
                words=self.text_norm(' '.join(w[0] for w in group)),
                start_time=offset + group[0][1] * self.embed_duration,
                end_time=offset + group[-1][2] * self.embed_duration,
            )

        for word, (start, end) in zip(words[1:], word_timestamps[1:]):
            if (start - group[-1][2]) * self.embed_duration > SEGMENT_SPLIT_GAP_SECONDS:
                segments.append(flush(group))
                group = [(word, start, end)]
            else:
                group.append((word, start, end))

        segments.append(flush(group))
        return segments

    def _speaker_label(self, speakers: List[str], spk_id: int) -> str:
        """Name the speaker a hypothesis is attributed to; see the `speaker_labels` argument.

        Out-of-range ids fall back to the slot name whatever the setting: the multi-talker model
        emits a fixed number of slots per session, more than some sessions have speakers, and
        those extra hypotheses still have to be scored (as insertions) rather than crash here.
        """
        if self.speaker_labels == 'slot' or spk_id >= len(speakers):
            return f'spk{spk_id}'
        return speakers[spk_id]

    def _hypothesis_segments(self, session_of, speakers_of, offset_of) -> Dict[int, List[SegLstSegment]]:
        """Build hypothesis STM segments from the accumulated states, keyed by manifest row id."""
        preds = dim_zero_cat(self.preds)
        preds_lengths = dim_zero_cat(self.preds_lengths)
        preds_word_timestamps = dim_zero_cat(self.preds_word_timestamps)
        preds_word_timestamps_lengths = dim_zero_cat(self.preds_word_timestamps_lengths)
        utt_ids = dim_zero_cat(self.utt_ids)
        spk_ids = dim_zero_cat(self.spk_ids)

        # When the dataset size is not divisible by the number of ranks, the distributed sampler
        # pads the last batch by repeating samples. Score each (session, speaker) pair once.
        already_processed_pairs = set()
        pred_segments = {}
        current_start = 0
        current_ts_start = 0

        for i in range(len(preds_lengths)):
            utt_id = utt_ids[i].item()
            spk_id = spk_ids[i].item()
            num_tokens = int(preds_lengths[i])
            num_word_timestamps = int(preds_word_timestamps_lengths[i])
            token_start, current_start = current_start, current_start + num_tokens
            ts_start, current_ts_start = current_ts_start, current_ts_start + num_word_timestamps

            if (utt_id, spk_id) in already_processed_pairs:
                continue
            already_processed_pairs.add((utt_id, spk_id))

            if utt_id not in session_of:
                logging.warning("Skipping prediction for unknown manifest row %s", utt_id)
                continue

            session_id = session_of[utt_id]
            speaker = self._speaker_label(speakers_of[utt_id], spk_id)
            offset = offset_of[utt_id]
            pred_segments.setdefault(utt_id, [])

            transcript = (
                self.decoding.decode_tokens_to_str(preds[token_start:current_start].detach().cpu())
                if num_tokens > 0
                else ''
            )
            words = transcript.split()

            if not words:
                # meeteval needs the (session, speaker) pair to exist even when nothing was said,
                # otherwise the deletions it should be charged for go uncounted.
                pred_segments[utt_id].append(
                    SegLstSegment(
                        session_id=session_id,
                        speaker=speaker,
                        words='',
                        start_time=offset,
                        end_time=offset + 1,
                    )
                )
                continue

            word_timestamps = preds_word_timestamps[ts_start:current_ts_start]
            if len(word_timestamps) != len(words):
                raise ValueError(
                    f"Number of word timestamps must match number of words: "
                    f"{len(word_timestamps)} != {len(words)}"
                )

            pred_segments[utt_id].extend(
                self._words_to_segments(words, word_timestamps.tolist(), session_id, speaker, offset)
            )

        return pred_segments

    def compute(self, targets_collection: List[Dict], save_stm_path: Optional[str] = None):
        """Score every accumulated hypothesis against the dataset's references.

        Args:
            targets_collection: The validation dataset's manifest collection, i.e.
                `dataloader.dataset.manifest_processor.collection`.
            save_stm_path: Directory to write `ref.stm` and `hyp.stm` into. Rank zero only.

        Returns:
            `(cp_result, tcp_result)` dicts with `wer`, `ins`, `del`, `sub` and `len`. When
            `compute_tcpwer` is False, `tcp_result` is `None` and tcpWER is never computed.
        """
        # Keyed by manifest row id (`sample.id`), which is what `update` records, and which is
        # not the position in the collection once empty sessions have been dropped.
        session_of = {sample.id: self._session_key(sample) for sample in targets_collection}
        speakers_of = {
            sample.id: sorted({segment['speaker'] for segment in sample.text_tokens})
            for sample in targets_collection
        }
        offset_of = {sample.id: (sample.offset or 0.0) for sample in targets_collection}

        gt_segments = self._reference_segments(targets_collection)
        pred_segments = self._hypothesis_segments(session_of, speakers_of, offset_of)

        # Every rank decodes only its shard of the dataset (`LazyDistributedSampler`), so no rank
        # holds enough to score a session on its own. Gather the hypotheses first; after this
        # every rank has all of them and the session-sharded scoring below is correct whatever
        # the decode split was.
        #
        # Keyed by manifest row id and merged into a dict, so this is also safe when the shards
        # overlap -- which they do, because the sampler pads the last shard by wrapping to keep
        # the per-rank batch counts equal. A row decoded on two ranks collapses to one entry
        # rather than being scored twice.
        world_size = get_world_size()
        if world_size > 1:
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, pred_segments)
            pred_segments = {row: segments for part in gathered for row, segments in part.items()}

        gt_segment_ids = sorted(gt_segments.keys())
        pred_segment_ids = sorted(pred_segments.keys())

        if len(gt_segment_ids) != len(pred_segment_ids):
            missing_rows = set(gt_segment_ids) - set(pred_segment_ids)
            missing_sessions = sorted({session_of.get(uid, uid) for uid in missing_rows})
            logging.warning(
                "References cover %d manifest rows but predictions cover %d; %d rows have no "
                "prediction, across recordings: %s%s. (Expected when limit_val_batches is set.)",
                len(gt_segment_ids),
                len(pred_segment_ids),
                len(missing_rows),
                missing_sessions[:10],
                f" and {len(missing_sessions) - 10} more" if len(missing_sessions) > 10 else "",
            )

        # Shard by SESSION, not by manifest row. cpWER optimizes a permutation per session, so
        # a session split across ranks is scored as two independent sessions -- strictly easier,
        # and dependent on the GPU count. That bit hard on segmented manifests: the AMI 30 s dev
        # set is 1585 windows over 18 recordings. Whole sessions keep the counts summable and
        # equal to scoring on one rank, which is what the ranks assume.
        sessions = sorted({session_of[uid] for uid in gt_segment_ids if uid in session_of})
        per_rank = ceil(len(sessions) / world_size) if sessions else 0
        begin_idx = get_rank() * per_rank
        mine = set(sessions[begin_idx:begin_idx + per_rank])

        gt_seg_lst = SegLST(
            segments=[
                seg for uid in gt_segment_ids if session_of.get(uid) in mine for seg in gt_segments[uid]
            ]
        )
        pred_seg_lst = SegLST(
            segments=[
                seg for uid in pred_segment_ids if session_of.get(uid) in mine for seg in pred_segments[uid]
            ]
        )

        if gt_seg_lst.segments:
            res_cp = self._process_metric_res(
                meeteval.wer.cpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst)
            )
            res_tcp = (
                self._process_metric_res(
                    meeteval.wer.tcpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst, collar=self.tcp_collar)
                )
                if self.compute_tcpwer
                else None
            )
        else:
            # Fewer rows than ranks leaves the last ranks with nothing to score, which meeteval
            # rejects as an empty reference. Zero counts are the right contribution to the sum.
            # Small per-dataset validation subsets hit this readily: 4 AMI dev cuts on 8 GPUs
            # gives half the ranks an empty shard.
            res_cp = self._process_metric_res({})
            res_tcp = self._process_metric_res({}) if self.compute_tcpwer else None

        res_both_all_ranks = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(res_both_all_ranks, (res_cp, res_tcp))
        else:
            res_both_all_ranks[0] = (res_cp, res_tcp)

        # `compute_tcpwer` comes from the config and so is identical on every rank; when it is off
        # each rank contributed `None` for the tcpWER half and there is nothing to pool.
        res_cp = pool_results([res[0] for res in res_both_all_ranks])
        res_tcp = pool_results([res[1] for res in res_both_all_ranks]) if self.compute_tcpwer else None

        if save_stm_path is not None and is_global_rank_zero():
            gt_seglist = SegLST(segments=[seg for uid in gt_segment_ids for seg in gt_segments[uid]])
            hyp_seglist = SegLST(segments=[seg for uid in pred_segment_ids for seg in pred_segments[uid]])
            meeteval.io.dump(gt_seglist, f'{save_stm_path}/ref.stm')
            meeteval.io.dump(hyp_seglist, f'{save_stm_path}/hyp.stm')

        return res_cp, res_tcp
