"""Precision report: a candidate and the baseline against an fp64 reference.

The reference is an fp64 run. The baseline is plain PyTorch on the same input, weights and
compute dtype. A mark sends a person to look at a row; it decides nothing.
"""

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

MARKED = (
    "abs_max",
    "abs_mean",
    "abs_median",
    "rel_max",
    "rel_mean",
    "rel_median",
    "cos_deficit",
)
CLEAR_ISSUE = 10.0
LARGE_DIFF = 2.0


def stats(out: Tensor, ref: Tensor) -> dict[str, float | bool]:
    """Cosine to `ref` and the max, mean and median of the abs and rel error of `out`."""
    flat_out = out.detach().double().flatten()
    flat_ref = ref.detach().double().flatten()

    # Absolute and relative error; relative only where the reference is nonzero
    diff = (flat_out - flat_ref).abs()
    nonzero = flat_ref != 0
    rel = (
        diff[nonzero] / flat_ref[nonzero].abs() if nonzero.any() else diff.new_zeros(1)
    )

    cos = 1.0
    if not flat_out.equal(flat_ref):
        cos = cosine_similarity(flat_out, flat_ref, dim=0).item()

    row = {
        "finite": bool(torch.isfinite(flat_out).all()),
        "cos": cos,
        "cos_deficit": 1.0 - cos,
    }
    for name, values in (("abs", diff), ("rel", rel)):
        row[f"{name}_max"] = values.max().item()
        row[f"{name}_mean"] = values.mean().item()
        row[f"{name}_median"] = values.median().item()
    return row


def compare(out: Tensor, baseline: Tensor, ref64: Tensor) -> dict[str, dict]:
    """The report row: `candidate` and `baseline` against `ref64`, and their ratios."""
    candidate = stats(out, ref64)
    plain = stats(baseline, ref64)
    ratios = {key: candidate[key] / max(plain[key], 1e-300) for key in MARKED}
    return {
        "candidate": candidate,
        "baseline": plain,
        "ratio": ratios,
        "mark": mark(candidate, plain, ratios),
    }


def mark(candidate: dict, baseline: dict, ratios: dict[str, float]) -> str | None:
    """`clear issue`, `large diff` or None, from how far the candidate sits past the baseline."""
    if baseline["finite"] and not candidate["finite"]:
        return "clear issue"
    worst = max(ratios[key] for key in MARKED)
    if worst >= CLEAR_ISSUE:
        return "clear issue"
    return "large diff" if worst >= LARGE_DIFF else None
