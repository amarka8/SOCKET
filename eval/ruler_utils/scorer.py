"""Thin scoring wrapper for RULER-32K.

The hub scores a whole results frame at once via ``calculate_metrics`` (per-task
``string_match``: ``string_match_part`` for ``qa_*`` tasks, ``string_match_all``
otherwise). We mirror that exactly: the eval loop accumulates one row per sample
into a results DataFrame, then this module scores the whole frame once. This is
what keeps the numbers paper-comparable.

Results-frame contract (one row per generated sample):
    - ``task``: the RULER task name (drives the metric choice).
    - ``predicted_answer``: the model's decoded string (special tokens stripped).
    - ``answer``: the list/ndarray of reference strings for that row (never
      collapsed to a single string).
"""

import pandas as pd

from eval.ruler_utils.calculate_metrics import calculate_metrics


def build_results_row(task, predicted_answer, answer):
    """Build a single results-frame row for accumulation in the eval loop.

    ``answer`` is kept as the list/ndarray of refs; ``predicted_answer`` is the
    decoded model output string.
    """
    return {
        "task": task,
        "predicted_answer": predicted_answer,
        "answer": answer,
    }


def ruler_score(results):
    """Score an accumulated RULER results frame.

    ``results`` may be a ``pandas.DataFrame`` or a list of row dicts produced by
    :func:`build_results_row`.

    Returns ``(per_task, overall)`` where ``per_task`` maps task -> string_match
    (float) and ``overall`` is the unweighted mean of the per-task scores.
    """
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(list(results))

    scores = calculate_metrics(results)  # {task: {"string_match": float}}

    per_task = {task: vals["string_match"] for task, vals in scores.items()}
    overall = (sum(per_task.values()) / len(per_task)) if per_task else 0.0
    return per_task, overall
