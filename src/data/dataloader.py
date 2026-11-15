"""Dataloader construction for the multi-talker STNO dataset.

Collapses NeMo's `get_audio_to_text_bpe_dataset_from_config` -> `get_bpe_dataset` ->
`EncDecRNNTBPEModel._setup_dataloader_from_config` chain into one function. All the branches
that chain carried (tarred, DALI, bucketing, concat, HuggingFace, code-switching) are
unreachable for this model and are not reproduced.
"""

from typing import Optional

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.utils import logging

from src.data.dataset import AudioToBPEAndSTNODataset
from src.data.dynamic_batching import DurationBucketBatchSampler, LazyDistributedSampler
from src.data.lhotse_dataset import LhotseToBPEAndSTNODataset
from src.data.lhotse_utils import manifest_paths

__all__ = ['get_stno_dataloader', 'uses_lhotse_manifest']

LHOTSE_SUFFIXES = ('.jsonl.gz', '.jsonl.gzip')


def uses_lhotse_manifest(config: DictConfig) -> bool:
    """Whether this dataset section points at a Lhotse CutSet rather than a NeMo manifest.

    An explicit `use_lhotse` always wins. Otherwise the format is inferred from the suffix of the
    first manifest, since a CutSet is conventionally `.jsonl.gz` while DiCoP's NeMo manifests are
    plain `.jsonl`. Set `use_lhotse: false` if you keep NeMo manifests gzipped.

    Several manifests may be given (as a list, or comma-separated), in which case they must all
    be the same format — one dataset reads them all.
    """
    if config.get('use_lhotse') is not None:
        return bool(config['use_lhotse'])

    manifest = config.get('manifest_filepath')
    if not manifest:
        return False

    paths = manifest_paths(manifest)
    is_lhotse = [path.endswith(LHOTSE_SUFFIXES) for path in paths]
    if any(is_lhotse) and not all(is_lhotse):
        raise ValueError(
            f"Mixed manifest formats in {list(paths)}: a dataset reads either Lhotse CutSets "
            f"({'/'.join(LHOTSE_SUFFIXES)}) or NeMo manifests, not both. Set `use_lhotse` "
            f"explicitly if the suffixes are misleading."
        )
    return is_lhotse[0]


def get_stno_dataloader(
    config: DictConfig,
    tokenizer,
    audio_downsampling_factor: int,
    text_norm_type: Optional[str] = 'whisper_nsf',
    val: bool = False,
    lightning_injects_sampler: bool = True,
    multi_speaker: bool = False,
    num_speaker_slots: int = 8,
    slot_order: str = 'first_occurrence',
    sot: bool = False,
    sot_order: str = 'speaking_time',
    sot_bos_id: Optional[int] = None,
    sot_eos_id: Optional[int] = None,
    sot_speaker_change_id: Optional[int] = None,
    group_speakers: bool = False,
) -> Optional[torch.utils.data.DataLoader]:
    """Build a dataloader over `(session, target speaker)` pairs.

    Args:
        config: One of the `model.{train,validation,test}_ds` sections.
        tokenizer: Tokenizer shared with the model's decoding object.
        audio_downsampling_factor: Samples per encoder frame (1280 for this architecture).
        text_norm_type: Text normalization applied before tokenization.
        val: Enumerate every (session, speaker) pair rather than sampling one speaker per
            session. Set for validation and test so scoring is deterministic and complete.
        multi_speaker: Emit whole sessions with one transcript per speaker slot instead, for the
            multi-talker model. Overrides `val`, which has nothing left to vary.
        num_speaker_slots: Speaker slots per item under `multi_speaker`.
        slot_order: How speakers are assigned to slots under `multi_speaker`.
        sot: Emit whole sessions with one *serialized* transcript covering every speaker
            instead, for the SOT model. Mutually exclusive with `multi_speaker`.
        sot_order: Which speaker ordering to serialize in under `sot`.
        sot_bos_id, sot_eos_id, sot_speaker_change_id: Token ids the serialized sequence is
            built with, required when `sot` is set.

    Returns:
        A `DataLoader`, or `None` if no manifest was configured.
    """
    if config.get('manifest_filepath') is None:
        return None

    augmentor = process_augmentations(config['augmentor']) if config.get('augmentor') is not None else None

    # Both datasets take the same arguments and emit the same 8-tuple; only the manifest
    # container differs.
    is_lhotse = uses_lhotse_manifest(config)
    dataset_cls = LhotseToBPEAndSTNODataset if is_lhotse else AudioToBPEAndSTNODataset
    logging.info(
        "Reading %s as a %s manifest", config['manifest_filepath'], "Lhotse CutSet" if is_lhotse else "NeMo"
    )

    dataset = dataset_cls(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        channel_selector=config.get('channel_selector', None),
        audio_downsampling_factor=audio_downsampling_factor,
        text_norm_type=config.get('txt_norm_type', text_norm_type),
        val=val,
        multi_speaker=multi_speaker,
        num_speaker_slots=num_speaker_slots,
        slot_order=slot_order,
        sot=sot,
        sot_order=sot_order,
        sot_bos_id=sot_bos_id,
        sot_eos_id=sot_eos_id,
        sot_speaker_change_id=sot_speaker_change_id,
        group_speakers=group_speakers,
        **({'stno_augment': config.get('stno_augment')} if is_lhotse else {}),
    )

    if len(dataset) == 0:
        logging.warning("Dataset built from %s is empty.", config['manifest_filepath'])

    num_workers = config.get('num_workers', 0)
    common = dict(
        dataset=dataset,
        collate_fn=dataset._collate_fn,
        num_workers=num_workers,
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', None) if num_workers > 0 else None,
        # Keep workers alive across epochs. Without this the pool is torn down and respawned every
        # epoch -- and, because validation runs at each epoch end, training resumes each time with
        # cold workers and an empty prefetch buffer, so the first ~num_workers*prefetch_factor steps
        # starve the GPU. The batch list (and its per-epoch reshuffle) is regenerated in the main
        # process via the sampler's set_epoch, so persistence does not freeze the shuffle. Requires
        # num_workers > 0. Off by default so other configs are unchanged.
        persistent_workers=bool(config.get('persistent_workers', False)) and num_workers > 0,
    )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1

    # Duration-based batching, for mixtures whose cuts differ in length by orders of magnitude.
    # Used for validation too when its config sets `max_batch_duration`: greedy decoding at a fixed
    # batch_size either wastes the card on short sessions or OOMs on a batch of long ones, because
    # memory tracks the padded (n_items x longest) rectangle. A duration budget keeps that rectangle
    # -- and the memory -- roughly constant across batches. For val the sampler runs with no weights
    # and no shuffle (see below), so `_build` enumerates every item exactly once: the decode stays
    # exhaustive and deterministic, and the sampler shards across ranks itself.
    max_batch_duration = config.get('max_batch_duration')
    if max_batch_duration and hasattr(dataset, 'item_durations'):
        # One weight per manifest, in `manifest_filepath` order, expanded to per-item weights.
        # Never weight-sample validation/test -- every item must be scored once.
        corpus_weights = None if val else config.get('corpus_weights')
        weights = None
        if corpus_weights:
            corpus_weights = list(corpus_weights)
            sources = dataset.item_sources
            if max(sources) + 1 > len(corpus_weights):
                raise ValueError(
                    f"corpus_weights has {len(corpus_weights)} entries for "
                    f"{max(sources) + 1} manifests"
                )
            weights = [float(corpus_weights[s]) for s in sources]
            logging.info("Corpus sampling weights: %s", corpus_weights)

        batch_sampler = DurationBucketBatchSampler(
            durations=dataset.item_durations,
            weights=weights,
            epoch_items=config.get('epoch_items'),
            max_batch_duration=float(max_batch_duration),
            max_batch_size=config.get('max_batch_size'),
            num_buckets=int(config.get('num_buckets', 10)),
            shuffle=config.get('shuffle', True),
            seed=int(config.get('seed', 0)),
            drop_last=config.get('drop_last', False),
            rank=rank,
            world_size=world_size,
        )
        return torch.utils.data.DataLoader(batch_sampler=batch_sampler, **common)

    # A custom batch sampler cannot be wrapped by Lightning's automatic sampler injection, so
    # runs that use one set `use_distributed_sampler: false` -- which also switches it off for
    # the evaluation loaders, which therefore have to be sharded here.
    # Lazy, because this runs inside `ModelPT.__init__` and the process group does not exist
    # yet -- an eagerly built `DistributedSampler` would capture world size 1 and every rank
    # would iterate the whole dataset. See `LazyDistributedSampler`.
    sampler = None
    if not lightning_injects_sampler:
        sampler = LazyDistributedSampler(dataset, shuffle=config.get('shuffle', False))

    return torch.utils.data.DataLoader(
        batch_size=config['batch_size'],
        drop_last=config.get('drop_last', False),
        shuffle=config.get('shuffle', False) if sampler is None else False,
        sampler=sampler,
        **common,
    )
