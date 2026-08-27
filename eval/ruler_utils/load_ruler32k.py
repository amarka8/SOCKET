"""Loader for the RULER-HARD-32K benchmark (xAlg-AI/att-hub-ruler-32k).

The hub dataset exposes one config/split per task; each split has exactly the
columns ``context, question, answer_prefix, answer, task, max_new_tokens`` and
200 rows. ``answer`` is a list/ndarray of reference strings (never collapse it
to a single string). We cap PER-SUBSET via ``.head(N)`` so the coverage stays
uniform across tasks rather than biasing a concatenated frame.
"""

import glob
import os

import pandas as pd
from datasets import load_dataset

RULER32K_HUB_NAME = "xAlg-AI/att-hub-ruler-32k"

# Columns we keep verbatim from the hub dataset.
RULER32K_COLUMNS = [
    "context",
    "question",
    "answer_prefix",
    "answer",
    "task",
    "max_new_tokens",
]


def _cache_arrow_to_df(subset: str) -> pd.DataFrame:
    """Read the subset's cached .arrow shard directly via pyarrow.

    Fallback used only when ``load_dataset`` cannot deserialize the cached
    ``dataset_info.json`` (a datasets-version/cache mismatch: the cache was
    written with a newer datasets that emits the ``List`` feature type, which an
    older datasets cannot reconstruct). The arrow row data itself is identical to
    what ``load_dataset`` would yield, so the resulting frame is paper-faithful.
    """
    import pyarrow as pa

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_name = RULER32K_HUB_NAME.replace("/", "___")
    pattern = os.path.join(
        hf_home, "datasets", cache_name, subset, "*", "*",
        f"att-hub-ruler-32k-{subset}.arrow",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cached arrow shard for RULER-32K subset '{subset}' under {pattern}"
        )
    path = matches[0]
    with pa.memory_map(path, "r") as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.lib.ArrowInvalid:
            src.seek(0)
            table = pa.ipc.open_file(src).read_all()
    return table.to_pandas()


def load_ruler32k_subset(subset: str, n: int = 100) -> pd.DataFrame:
    """Load a single RULER-32K subset (config == split == task name).

    Returns a pandas DataFrame capped to the first ``n`` rows, keeping the
    canonical columns. ``answer`` is preserved as a list/ndarray of refs.

    Primary path is ``load_dataset`` (paper-comparable). If that fails on a
    datasets-version/cache mismatch, falls back to reading the cached arrow
    shard directly (identical row data).
    """
    try:
        ds = load_dataset(RULER32K_HUB_NAME, subset, split=subset)
        df = ds.to_pandas()
    except Exception:
        df = _cache_arrow_to_df(subset)

    df = df.head(n)
    # Keep only the canonical columns (and only those that exist).
    keep = [c for c in RULER32K_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


def ruler_num_samples(default: int = 100) -> int:
    """Per-task sample cap, overridable via the ``RULER_NUM_SAMPLES`` env var.

    Lets the smoke test run a handful of rows without editing config/code;
    leaving the var unset gets the 100-row default.
    """
    val = os.environ.get("RULER_NUM_SAMPLES")
    if val is None or val.strip() == "":
        return default
    try:
        n = int(val)
    except ValueError:
        return default
    return n if n > 0 else default


def load_ruler32k(subset: str, n: int = 100) -> pd.DataFrame:
    """Public entry point. ``subset`` is a single RULER task name.

    The effective cap honors ``RULER_NUM_SAMPLES`` (falls back to ``n``).
    """
    return load_ruler32k_subset(subset, n=ruler_num_samples(default=n))
