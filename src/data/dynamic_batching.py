"""Batch by padded audio duration rather than by a fixed count.

A fixed `batch_size` has to be chosen for the longest cut in the mixture. Training LibriMix (~10 s)
alongside NOTSOFAR sessions (~360 s) and AMI windows (600 s) then wastes almost all of the card on
the short corpora: batch 2 of AMI fills 79 GiB, batch 2 of LibriMix a rounding error of that.

What decides memory is the *padded* batch -- every item is padded to the longest, so the cost is
`len(batch) * max_duration(batch)`, not the sum of durations. This sampler fills each batch up to
a budget on that product, and groups items of similar length into buckets first so the padding
inside a batch stays small.

Sharding is done here rather than by Lightning: a custom batch sampler cannot be wrapped by its
automatic `DistributedSampler` injection, so runs using this need `use_distributed_sampler: false`
and get their per-rank split from `rank`/`world_size` below. Every rank builds the identical batch
list from the same seed and epoch, then takes a strided slice of it, and all ranks are truncated
to the same number of batches -- an uneven count would hang DDP at the end of an epoch.
"""

import logging
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler

__all__ = ['DurationBucketBatchSampler', 'LazyDistributedSampler']


class DurationBucketBatchSampler(Sampler[List[int]]):
    """Yields lists of dataset indices whose padded duration fits a budget.

    Args:
        durations: Duration in seconds of every dataset item, indexed as the dataset is.
        max_batch_duration: Budget in seconds for `len(batch) * max(duration in batch)`.
        max_batch_size: Hard cap on items per batch, whatever the budget allows.
        num_buckets: Length buckets to draw batches from. More buckets means less padding and
            less mixing of corpora within a batch.
        shuffle: Shuffle within buckets and across batches, reseeded every epoch.
        drop_last: Drop a bucket's final, partly filled batch.
        rank, world_size: This rank's share of the batches.
    """

    def __init__(
        self,
        durations: Sequence[float],
        max_batch_duration: float,
        max_batch_size: Optional[int] = None,
        num_buckets: int = 10,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        weights: Optional[Sequence[float]] = None,
        epoch_items: Optional[int] = None,
    ):
        if max_batch_duration <= 0:
            raise ValueError(f"max_batch_duration must be positive, got {max_batch_duration}")
        longest = max(durations) if len(durations) else 0.0
        if longest > max_batch_duration:
            raise ValueError(
                f"max_batch_duration ({max_batch_duration:.0f}s) is below the longest item "
                f"({longest:.0f}s), so that item could never be batched."
            )

        self.durations = list(durations)
        self.max_batch_duration = float(max_batch_duration)
        self.max_batch_size = max_batch_size
        self.num_buckets = max(1, num_buckets)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._rank = rank
        self._world_size = max(1, world_size)
        self.epoch = 0
        self._cache_key = None
        self._cache: List[List[int]] = []

        # Per-item sampling weights, for giving some corpora a higher prior than their item count
        # would give them. An epoch then draws `epoch_items` items in proportion to the weights
        # (with replacement), rather than enumerating every item once: a corpus weighted 10x is
        # seen about ten times as often per item, whether that means repeating a small corpus or
        # visiting only part of a large one each epoch.
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.double)
        if self.weights is not None:
            if len(self.weights) != len(self.durations):
                raise ValueError(
                    f"weights has {len(self.weights)} entries for {len(self.durations)} items"
                )
            if float(self.weights.sum()) <= 0:
                raise ValueError("weights must contain at least one positive entry")
        self.epoch_items = epoch_items or len(self.durations)

    def _ddp(self):
        """Rank and world size, resolved late.

        NeMo builds the dataloaders in `ModelPT.__init__`, before the process group exists, so
        reading them in the constructor would shard every rank against a world size of 1 and
        train each on the whole dataset.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return self._rank, self._world_size

    @property
    def micro_batch_size(self) -> int:
        """Average items per batch. NeMo reads this off a batch sampler to estimate max_steps."""
        batches = self._batches()
        return max(1, round(len(self.durations) / max(1, len(batches) * self._ddp()[1])))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        rank, world_size = self._ddp()
        key = (self.epoch, rank, world_size)
        if key != self._cache_key:
            self._cache = self._build(self.epoch, rank, world_size)
            self._cache_key = key
            sizes = [len(b) for b in self._cache]
            logging.info(
                "Dynamic batching: %d batches for rank %d of %d, %d-%d items each (median %d), "
                "budget %.0f s of padded audio",
                len(sizes), rank, world_size, min(sizes) if sizes else 0, max(sizes) if sizes else 0,
                sorted(sizes)[len(sizes) // 2] if sizes else 0, self.max_batch_duration,
            )
        return self._cache

    def _build(self, epoch: int, rank: int, world_size: int) -> List[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        if self.weights is None:
            pool = list(range(len(self.durations)))
        else:
            pool = torch.multinomial(
                self.weights, self.epoch_items, replacement=True, generator=generator
            ).tolist()

        order = sorted(pool, key=lambda i: self.durations[i])
        bucket_size = max(1, len(order) // self.num_buckets)
        buckets = [order[i:i + bucket_size] for i in range(0, len(order), bucket_size)]

        batches: List[List[int]] = []
        for bucket in buckets:
            if self.shuffle:
                bucket = [bucket[i] for i in torch.randperm(len(bucket), generator=generator)]
            batch: List[int] = []
            longest = 0.0
            for index in bucket:
                candidate = max(longest, self.durations[index])
                over_budget = (len(batch) + 1) * candidate > self.max_batch_duration
                over_count = self.max_batch_size is not None and len(batch) + 1 > self.max_batch_size
                if batch and (over_budget or over_count):
                    batches.append(batch)
                    batch, longest = [index], self.durations[index]
                else:
                    batch.append(index)
                    longest = candidate
            if batch and not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator)]

        # Equal work per rank: DDP hangs if one rank runs out of batches before the others.
        per_rank = len(batches) // world_size
        return [batches[i * world_size + rank] for i in range(per_rank)]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())

class LazyDistributedSampler(torch.utils.data.Sampler):
    """Shard a dataset across ranks, resolving the rank at ITERATION time.

    `torch.utils.data.DistributedSampler` takes `rank` and `num_replicas` in its constructor, and
    NeMo builds the evaluation dataloaders inside `ModelPT.__init__` -- before Lightning has
    initialized the process group. Constructed there, `torch.distributed.is_initialized()` is
    still False, so every rank believes it is rank 0 of a world of 1 and iterates the WHOLE
    dataset. That is not a crash, it is silent duplicated work: on the SOT run, both ranks
    decoded all 1585 AMI dev windows, and validation was 34% of the epoch with half of it
    redundant. `DurationBucketBatchSampler` already dodges this by reading rank and world size
    lazily; this is the same trick for the plain evaluation loaders.

    Indices are padded by wrapping so every rank yields the same number of batches -- an uneven
    split leaves ranks running different numbers of steps, which deadlocks the first collective
    that assumes otherwise. The duplicates that padding introduces are harmless here because the
    metric merges hypotheses by manifest row id, so a row decoded twice collapses to one entry.
    """

    def __init__(self, dataset, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Called by Lightning between epochs; only matters when `shuffle` is on."""
        self.epoch = epoch

    def _indices(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        count = len(self.dataset)
        order = list(range(count))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = [order[i] for i in torch.randperm(count, generator=generator).tolist()]

        per_rank = -(-count // world_size)
        order = order + order[: per_rank * world_size - count]
        return order[rank::world_size]

    def __iter__(self):
        return iter(self._indices())

    def __len__(self):
        return len(self._indices())
