"""S5: the ported fixed losses against the 0.3.13 record from s5_record.

Per case: the new loss at fp64 must reproduce the old fp64 value and input gradient
(identity), and at fp32 and bf16 its precision row is compared to the old one at the
same dtype, both against the fp64 record. Also checks the registry flags map one to one.
Run s5_record first, then plain `python measure/s5_fixed.py`.
"""

import sys

sys.path.insert(0, "/xg7/SenseCraft/src")

import json
from pathlib import Path

import torch
from stats import compare, stats

import sensecraft.pixel
import sensecraft.signal  # noqa: F401
from sensecraft.registry import TERMS

SEED = 0
REF_PATH = Path(__file__).parent / "refs" / "s5_fixed.pt"
OUT_PATH = Path(__file__).parent / "refs" / "s5_fixed.jsonl"
IDENTITY_TOLERANCE = 1e-12  # rel_max the fp64 port may differ from the fp64 record

# Constructor and forward kwargs of the new API for the record's non-default cases
NEW_KWARGS = {"gaussian_noise+noise": {"sigma_range": (0.05, 0.2), "add_noise": True}}
# Cases that differ from 0.3.13 on purpose, and why; their differences are reported, not failed
CHANGED = {
    "tssim": "registered as [-1, 1], the range its formulas assume",
    "ssim3d+symmetric": "[-1, 1] clips are no longer guessed from the minimum and remapped",
    "stssim+symmetric": "[-1, 1] clips are no longer guessed from the minimum and remapped",
}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
SEVERITY = {None: 0, "large diff": 1, "clear issue": 2}


def run(loss, input, target, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Value and input gradient of one forward and backward at `dtype`."""
    input = input.detach().to(dtype).clone().requires_grad_(True)
    torch.manual_seed(SEED)
    value = loss(input, target.to(dtype))
    value.backward()
    return value.detach(), input.grad


def flags_match(term, old_flags: dict) -> bool:
    """The new registration carries the old range, layout and fp32 flags."""
    dims = "2d" if old_flags["is_2d_only"] else "any"
    dims = "3d" if old_flags["is_3d_only"] else dims
    return (
        term.range.value == old_flags["range"]
        and term.dims == dims
        and term.fp32 == old_flags["requires_fp32"]
    )


def worst(*marks: str | None) -> str | None:
    return max(marks, key=SEVERITY.__getitem__)


def main():
    record = torch.load(REF_PATH)
    rows = []
    identity_failures = []

    for case, entry in record.items():
        term = TERMS[entry["name"]]
        loss = term.factory(**NEW_KWARGS.get(case, entry["kwargs"]))
        input, target = entry["input"], entry["target"]
        old_runs = entry["runs"]
        ref_value, ref_grad = old_runs["float64"]

        # Identity: new fp64 against the old fp64 record
        value, grad = run(loss, input, target, torch.float64)
        identity = {
            "value_rel": stats(value, ref_value)["rel_max"],
            "grad_rel": stats(grad, ref_grad)["rel_max"],
            "grad_abs": stats(grad, ref_grad)["abs_max"],
        }
        identical = (
            max(identity["value_rel"], identity["grad_rel"]) <= IDENTITY_TOLERANCE
        )
        flags_ok = flags_match(term, entry["flags"])
        if case in CHANGED:
            print(f"{case:22s} changed on purpose: {CHANGED[case]}")
            continue
        if not (identical and flags_ok):
            identity_failures.append(case)
        print(
            f"{case:22s} float64  identity {'ok ' if identical else 'BAD'}"
            f" flags {'ok ' if flags_ok else 'BAD'}"
            f" value_rel {identity['value_rel']:.1e}"
            f" grad_rel {identity['grad_rel']:.1e}"
        )
        rows.append(
            {"case": case, "dtype": "float64", **identity, "flags_ok": flags_ok}
        )

        # Precision: new and old at the same dtype, both against the fp64 record
        for dtype_name, dtype in DTYPES.items():
            if dtype_name not in old_runs:
                continue
            value, grad = run(loss, input, target, dtype)
            old_value, old_grad = old_runs[dtype_name]
            value_row = compare(value, old_value, ref_value)
            grad_row = compare(grad, old_grad, ref_grad)
            mark = worst(value_row["mark"], grad_row["mark"])
            print(
                f"{case:22s} {dtype_name:8s} {mark or '-'!s:11s}"
                f" value_rel {value_row['candidate']['rel_max']:.2e}"
                f" (old {value_row['baseline']['rel_max']:.2e})"
                f" grad_rel_median {grad_row['candidate']['rel_median']:.2e}"
                f" (old {grad_row['baseline']['rel_median']:.2e})"
                f" cos {grad_row['candidate']['cos']:.8f}"
            )
            rows.append(
                {
                    "case": case,
                    "dtype": dtype_name,
                    "mark": mark,
                    "value": value_row,
                    "grad": grad_row,
                }
            )

    with OUT_PATH.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    marks = [row["mark"] for row in rows if row["dtype"] != "float64"]
    counts = {str(mark): marks.count(mark) for mark in SEVERITY}
    print(f"identity failures: {identity_failures or 'none'}; marks: {counts}")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
