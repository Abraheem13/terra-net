"""Fold-level parallelism for the leave-one-city-out scripts.

The folds of every script are independent, so `map_folds` runs them in
forked worker processes (the loaded corpus is shared copy-on-write) with the
BLAS/OpenMP/torch thread pools of every worker limited to one thread. The
results do not depend on the number of workers: every fold seeds its own
random generators, and the outputs are concatenated in fold order.

The number of workers comes from --jobs, or from the environment variable
TERRANET_JOBS (scripts/00_reproduce.py sets it), default 1.
"""
from __future__ import annotations

import os
from multiprocessing import get_context


def n_jobs(default: int = 1) -> int:
    return max(1, int(os.environ.get("TERRANET_JOBS", default)))


def _init_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def map_folds(fn, items, jobs: int):
    """[fn(item) for item in items], in order, on `jobs` forked workers."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    ctx = get_context("fork")
    with ctx.Pool(min(jobs, len(items)), initializer=_init_worker, initargs=(1,)) as pool:
        return pool.map(fn, items, chunksize=1)
