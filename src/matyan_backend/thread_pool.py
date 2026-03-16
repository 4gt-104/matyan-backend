"""Dedicated thread pool for FoundationDB I/O.

All FDB-touching work (storage reads, bundle fetches, sequence sampling,
query iterators) runs on :data:`FDB_EXECUTOR` instead of the default
``ThreadPoolExecutor`` so that zombie tasks from cancelled streaming
responses cannot starve unrelated endpoints.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from matyan_backend.config import SETTINGS

FDB_EXECUTOR = ThreadPoolExecutor(
    max_workers=SETTINGS.fdb_thread_pool_size,
    thread_name_prefix="fdb-io",
)


async def to_fdb_thread(fn: Any, /, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Run *fn* on the dedicated FDB thread pool.

    Drop-in replacement for :func:`asyncio.to_thread` that uses
    :data:`FDB_EXECUTOR` instead of the default executor.

    :param fn: Callable to execute in a worker thread.
    :param args: Positional arguments forwarded to *fn*.
    :param kwargs: Keyword arguments forwarded to *fn*.
    :returns: The return value of *fn*.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        fn = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(FDB_EXECUTOR, fn)
    return await loop.run_in_executor(FDB_EXECUTOR, fn, *args)
