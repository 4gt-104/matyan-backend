"""Time series data storage: write/read/sample metric sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matyan_backend.fdb_types import transactional

from . import encoding
from .fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import DirectorySubspace, Transaction


def _runs_dir() -> DirectorySubspace:
    return get_directories().runs


def _empty_result(columns: tuple[str, ...]) -> dict[str, list]:
    result: dict[str, list] = {"steps": []}
    for col in columns:
        result[col] = []
    return result


def _pick_evenly_spaced(items: list, num_points: int) -> list:
    """Return up to *num_points* evenly-spaced elements from *items*."""
    total = len(items)
    if total <= num_points:
        return items
    if num_points <= 1:
        return [items[0]]
    step_size = (total - 1) / (num_points - 1)
    indices = list(dict.fromkeys(round(i * step_size) for i in range(num_points)))
    return [items[idx] for idx in indices]


def _evenly_spaced_targets(first: int, last: int, num_points: int) -> list[int]:
    """Compute *num_points* evenly-spaced integer targets in ``[first, last]``."""
    if num_points <= 1:
        return [first]
    span = last - first
    return list(dict.fromkeys(
        first + round(i * span / (num_points - 1)) for i in range(num_points)
    ))


def _read_extra_columns(
    tr: Transaction,
    rd: DirectorySubspace,
    run_hash: str,
    ctx_id: int,
    name: str,
    steps: list[int],
    columns: tuple[str, ...],
) -> dict[str, list]:
    """Fetch non-val columns for the given steps."""
    result: dict[str, list] = {}
    for col in columns:
        vals: list = []
        for step in steps:
            raw = tr[rd.pack((run_hash, "seqs", ctx_id, name, col, step))]
            vals.append(encoding.decode_value(raw) if raw.present() else None)
        result[col] = vals
    return result


_STREAM_SCAN_NUM_POINTS_THRESHOLD = 128


def _point_read_sample(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    num_points: int,
    range_begin: bytes,
    range_end: bytes,
    columns: tuple[str, ...],
) -> dict[str, list]:
    """Sample *num_points* from a range using bounded forward scans.

    Each target step is resolved to the nearest existing step >= target
    via ``get_range(begin, range_end, limit=1)``.  Total FDB reads are
    O(num_points) instead of O(N).

    If point reads return fewer than *num_points* results (sparse steps
    with uneven distribution), the sequence is small enough that a full
    range scan is cheap — fall back to ``_pick_evenly_spaced`` on the
    full result set.
    """
    rd = _runs_dir()
    val_base = (run_hash, "seqs", ctx_id, name, "val")

    first_kv = next(iter(tr.get_range(range_begin, range_end, limit=1)), None)
    if first_kv is None:
        return _empty_result(columns)

    last_kv = next(
        iter(tr.get_range(range_begin, range_end, limit=1, reverse=True)),
        None,
    )
    first_step: int = rd.unpack(first_kv.key)[-1]
    last_step: int = rd.unpack(last_kv.key)[-1] if last_kv else first_step

    result = _empty_result(columns)
    if first_step == last_step or num_points <= 1:
        result["steps"].append(first_step)
        if "val" in columns:
            result["val"].append(encoding.decode_value(first_kv.value))
    else:
        targets = _evenly_spaced_targets(first_step, last_step, num_points)
        seen: set[int] = set()
        for target in targets:
            begin = rd.pack((*val_base, target))
            kv = next(iter(tr.get_range(begin, range_end, limit=1)), None)
            if kv is None:
                continue
            step = rd.unpack(kv.key)[-1]
            if step in seen:
                continue
            seen.add(step)
            result["steps"].append(step)
            if "val" in columns:
                result["val"].append(encoding.decode_value(kv.value))

        if len(result["steps"]) < num_points:
            return _full_scan_sample(
                tr, rd, run_hash, ctx_id, name, num_points,
                range_begin, range_end, columns,
            )

    extra_cols = tuple(c for c in columns if c != "val")
    if extra_cols:
        result.update(
            _read_extra_columns(tr, rd, run_hash, ctx_id, name, result["steps"], extra_cols),
        )
    return result


def _stream_scan_sample(  # noqa: C901, PLR0912
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    num_points: int,
    range_begin: bytes,
    range_end: bytes,
    columns: tuple[str, ...],
) -> dict[str, list]:
    """Sample a range with one forward scan and O(num_points) memory.

    Preserves the first and last points and chooses at most one deterministic
    representative for each interior logical bucket.
    """
    rd = _runs_dir()
    first_kv = next(iter(tr.get_range(range_begin, range_end, limit=1)), None)
    if first_kv is None:
        return _empty_result(columns)

    last_kv = next(
        iter(tr.get_range(range_begin, range_end, limit=1, reverse=True)),
        None,
    )
    first_step: int = rd.unpack(first_kv.key)[-1]
    last_step: int = rd.unpack(last_kv.key)[-1] if last_kv else first_step

    result = _empty_result(columns)
    if first_step == last_step or num_points <= 1:
        result["steps"].append(first_step)
        if "val" in columns:
            result["val"].append(encoding.decode_value(first_kv.value))
        extra_cols = tuple(c for c in columns if c != "val")
        if extra_cols:
            result.update(_read_extra_columns(tr, rd, run_hash, ctx_id, name, result["steps"], extra_cols))
        return result

    bucket_count = max(0, num_points - 2)
    bucket_span = max(1, last_step - first_step - 1)
    buckets: list[tuple[int, Any] | None] = [None] * bucket_count

    first_val = encoding.decode_value(first_kv.value) if "val" in columns else None
    last_val = encoding.decode_value(last_kv.value) if ("val" in columns and last_kv is not None) else first_val

    if bucket_count > 0:
        for kv in tr.get_range(range_begin, range_end):
            step = rd.unpack(kv.key)[-1]
            if step in (first_step, last_step):
                continue
            bucket_idx = min(bucket_count - 1, ((step - first_step - 1) * bucket_count) // bucket_span)
            if buckets[bucket_idx] is None:
                val = encoding.decode_value(kv.value) if "val" in columns else None
                buckets[bucket_idx] = (step, val)

    result["steps"].append(first_step)
    if "val" in columns:
        result["val"].append(first_val)

    for bucket in buckets:
        if bucket is None:
            continue
        step, val = bucket
        result["steps"].append(step)
        if "val" in columns:
            result["val"].append(val)

    if result["steps"][-1] != last_step:
        result["steps"].append(last_step)
        if "val" in columns:
            result["val"].append(last_val)

    extra_cols = tuple(c for c in columns if c != "val")
    if extra_cols:
        result.update(_read_extra_columns(tr, rd, run_hash, ctx_id, name, result["steps"], extra_cols))
    return result


def _full_scan_sample(
    tr: Transaction,
    rd: DirectorySubspace,
    run_hash: str,
    ctx_id: int,
    name: str,
    num_points: int,
    range_begin: bytes,
    range_end: bytes,
    columns: tuple[str, ...],
) -> dict[str, list]:
    """Fallback: full range scan + evenly-spaced pick.

    Only called when there are fewer steps than *num_points*, so the
    scan is cheap by definition.
    """
    all_kvs = list(tr.get_range(range_begin, range_end))
    if not all_kvs:
        return _empty_result(columns)

    sampled = _pick_evenly_spaced(all_kvs, num_points)
    result = _empty_result(columns)
    for kv in sampled:
        step = rd.unpack(kv.key)[-1]
        result["steps"].append(step)
        if "val" in columns:
            result["val"].append(encoding.decode_value(kv.value))

    extra_cols = tuple(c for c in columns if c != "val")
    if extra_cols:
        result.update(
            _read_extra_columns(tr, rd, run_hash, ctx_id, name, result["steps"], extra_cols),
        )
    return result


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


@transactional
def write_sequence_step(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    step: int,
    value: Any,  # noqa: ANN401
    epoch: int | None = None,
    timestamp: float | None = None,
) -> None:
    rd = _runs_dir()
    base = (run_hash, "seqs", ctx_id, name)
    tr[rd.pack((*base, "val", step))] = encoding.encode_value(value)
    tr[rd.pack((*base, "step", step))] = encoding.encode_value(step)
    if epoch is not None:
        tr[rd.pack((*base, "epoch", step))] = encoding.encode_value(epoch)
    if timestamp is not None:
        tr[rd.pack((*base, "time", step))] = encoding.encode_value(timestamp)


@transactional
def write_sequence_batch(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    steps_data: list[dict],
) -> None:
    """Write multiple steps in a single transaction.

    Each dict in *steps_data* must have ``step`` and ``value``, and may have
    ``epoch`` and ``timestamp``.
    """
    rd = _runs_dir()
    base = (run_hash, "seqs", ctx_id, name)
    for entry in steps_data:
        step = entry["step"]
        tr[rd.pack((*base, "val", step))] = encoding.encode_value(entry["value"])
        tr[rd.pack((*base, "step", step))] = encoding.encode_value(step)
        if "epoch" in entry and entry["epoch"] is not None:
            tr[rd.pack((*base, "epoch", step))] = encoding.encode_value(entry["epoch"])
        if "timestamp" in entry and entry["timestamp"] is not None:
            tr[rd.pack((*base, "time", step))] = encoding.encode_value(entry["timestamp"])


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@transactional
def read_sequence(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
    columns: tuple[str, ...] = ("val",),
) -> dict[str, list]:
    """Read sequence data, optionally bounded by step range.

    Returns ``{column: [values...]}`` plus a special ``"steps"`` key with
    the step integers.
    """
    rd = _runs_dir()
    result = _empty_result(columns)

    val_base = (run_hash, "seqs", ctx_id, name, "val")
    full_range = rd.range(val_base)
    begin = rd.pack((*val_base, start_step)) if start_step is not None else full_range.start
    end = min(rd.pack((*val_base, end_step + 1)), full_range.stop) if end_step is not None else full_range.stop

    for kv in tr.get_range(begin, end):
        step = rd.unpack(kv.key)[-1]
        result["steps"].append(step)
        if "val" in columns:
            result["val"].append(encoding.decode_value(kv.value))

    extra_cols = tuple(c for c in columns if c != "val")
    if extra_cols:
        result.update(_read_extra_columns(tr, rd, run_hash, ctx_id, name, result["steps"], extra_cols))

    return result


@transactional
def sample_sequence(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    num_points: int,
    *,
    columns: tuple[str, ...] = ("val",),
) -> dict[str, list]:
    """Uniformly sample *num_points* from a sequence.

    Uses O(num_points) bounded range scans instead of materializing the
    entire sequence.  For each evenly-spaced target step the nearest
    existing step >= target is found via ``get_range(limit=1)``.
    """
    rd = _runs_dir()
    full_range = rd.range((run_hash, "seqs", ctx_id, name, "val"))
    return _point_read_sample(
        tr, run_hash, ctx_id, name, num_points,
        full_range.start, full_range.stop, columns,
    )


@transactional
def read_and_sample_sequence(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
    density: int | None = None,
    columns: tuple[str, ...] = ("val",),
) -> dict[str, list]:
    """Read a sequence with optional range filtering and downsampling.

    When *density* is set, uses O(density) point reads instead of
    materializing the entire range.
    """
    if not density or density <= 0:
        return read_sequence(
            tr, run_hash, ctx_id, name,
            start_step=start_step, end_step=end_step, columns=columns,
        )

    rd = _runs_dir()
    val_base = (run_hash, "seqs", ctx_id, name, "val")
    full_range = rd.range(val_base)
    range_begin = rd.pack((*val_base, start_step)) if start_step is not None else full_range.start
    range_end = (
        min(rd.pack((*val_base, end_step + 1)), full_range.stop)
        if end_step is not None
        else full_range.stop
    )
    return _point_read_sample(
        tr, run_hash, ctx_id, name, density,
        range_begin, range_end, columns,
    )


@transactional
def sample_sequences_batch(
    tr: Transaction,
    run_hash: str,
    requests: list[tuple[int, str]],
    num_points: int,
    *,
    columns: tuple[str, ...] = ("val", "epoch", "time"),
    x_axis_name: str | None = None,
) -> tuple[dict[tuple[int, str], dict[str, list]], dict[int, dict[str, list]] | None]:
    """Sample multiple sequences for one run in a single FDB transaction.

    Returns ``(main_results, x_axis_results)``:
    - *main_results*: ``{(ctx_id, name): {steps, val, ...}}`` for each request.
    - *x_axis_results*: ``{ctx_id: {steps, val}}`` if *x_axis_name* is set,
      else ``None``.
    """
    rd = _runs_dir()
    main: dict[tuple[int, str], dict[str, list]] = {}

    for ctx_id, name in dict.fromkeys(requests):
        full_range = rd.range((run_hash, "seqs", ctx_id, name, "val"))
        sample_fn = _stream_scan_sample if num_points >= _STREAM_SCAN_NUM_POINTS_THRESHOLD else _point_read_sample
        main[(ctx_id, name)] = sample_fn(
            tr,
            run_hash,
            ctx_id,
            name,
            num_points,
            full_range.start,
            full_range.stop,
            columns,
        )

    x_axis: dict[int, dict[str, list]] | None = None
    if x_axis_name is not None:
        x_axis = {}
        seen_ctx: set[int] = set()
        for ctx_id, _ in requests:
            if ctx_id in seen_ctx:
                continue
            seen_ctx.add(ctx_id)
            full_range = rd.range((run_hash, "seqs", ctx_id, x_axis_name, "val"))
            sample_fn = _stream_scan_sample if num_points >= _STREAM_SCAN_NUM_POINTS_THRESHOLD else _point_read_sample
            x_axis[ctx_id] = sample_fn(
                tr,
                run_hash,
                ctx_id,
                x_axis_name,
                num_points,
                full_range.start,
                full_range.stop,
                ("val",),
            )

    return main, x_axis


@transactional
def get_sequence_step_bounds(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
) -> tuple[int | None, int | None]:
    """Return ``(first_step, last_step)`` for a sequence, or ``(None, None)`` if empty."""
    rd = _runs_dir()
    val_base = (run_hash, "seqs", ctx_id, name, "val")
    r = rd.range(val_base)

    first_kv = next(iter(tr.get_range(r.start, r.stop, limit=1)), None)
    if first_kv is None:
        return (None, None)
    last_kv = next(iter(tr.get_range(r.start, r.stop, limit=1, reverse=True)), None)
    first_step: int = rd.unpack(first_kv.key)[-1]
    last_step: int = rd.unpack(last_kv.key)[-1] if last_kv else first_step
    return (first_step, last_step)


@transactional
def get_sequence_last_step(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
) -> int | None:
    """Return the highest step number in a sequence, or ``None`` if empty.

    Uses a reverse range scan — O(1) regardless of sequence size.
    """
    rd = _runs_dir()
    r = rd.range((run_hash, "seqs", ctx_id, name, "val"))
    last_kv = next(iter(tr.get_range(r.start, r.stop, limit=1, reverse=True)), None)
    if last_kv is None:
        return None
    return rd.unpack(last_kv.key)[-1]


@transactional
def get_sequence_length(tr: Transaction, run_hash: str, ctx_id: int, name: str) -> int:
    rd = _runs_dir()
    r = rd.range((run_hash, "seqs", ctx_id, name, "val"))
    count = 0
    for _ in tr.get_range(r.start, r.stop):
        count += 1
    return count
