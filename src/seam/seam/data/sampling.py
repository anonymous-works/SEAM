from __future__ import annotations

import math
import random
from collections.abc import Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 42, multiplier: int = 50):
        if batch_size <= 0 or multiplier <= 0:
            raise ValueError("Batch size and bucket multiplier must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.multiplier = multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        pool_size = self.batch_size * self.multiplier
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start : start + pool_size], key=self.lengths.__getitem__)
            batches.extend(pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        yield from batches


class DistributedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        num_replicas: int,
        rank: int,
        *,
        seed: int = 42,
        multiplier: int = 1,
        shuffle: bool = True,
    ):
        if batch_size <= 0 or num_replicas <= 0:
            raise ValueError("Batch size and num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must lie in [0, num_replicas)")
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.multiplier = int(multiplier)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        pool_size = self.batch_size * self.multiplier
        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = indices[start : start + pool_size]
            if self.multiplier > 1:
                pool = sorted(pool, key=self.lengths.__getitem__)
            batches.extend(pool[offset : offset + self.batch_size] for offset in range(0, len(pool), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __len__(self) -> int:
        batches = math.ceil(len(self.lengths) / self.batch_size)
        return math.ceil(batches / self.num_replicas)

    def __iter__(self):
        batches = self._batches()
        if not batches:
            return
        local_batch_count = len(self)
        total_batch_count = local_batch_count * self.num_replicas
        if len(batches) < total_batch_count:
            batches.extend(batches[index % len(batches)] for index in range(total_batch_count - len(batches)))
        yield from batches[self.rank : total_batch_count : self.num_replicas]
