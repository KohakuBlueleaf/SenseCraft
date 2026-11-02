"""Stages S0 to S2 for the perceptual terms: weights, fp64 identity, then each dtype.

Reads `refs/s1_<key>.pt` from `s1_record.py`. S0: every converted weight equals one of upstream's,
tensor for tensor. S1: our net at fp64 against upstream at fp64. S2: our net at each dtype (candidate)
against upstream at that dtype (baseline), both against upstream fp64. One forward and backward
per row. Real compute.

    python measure/s2_nets.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from common import DTYPES, REFS, cotangents, digest, input_gradient
from stats import compare, stats
from terms import TERMS, ours

DEVICE: str = "cuda:0"
FP16_SCALE: float = 2.0**14  # the loss scale an fp16 run backs the value through
KEYS: list[str] = list(TERMS)
CHANNELS_LAST: bool = True  # the layout fused kernels need
OUT: Path = REFS / "s2_nets.jsonl"


def rows_of(name: str, ref: dict, term, features_of, dtype_name: str) -> list[dict]:
    """Precision rows of one term at one dtype: every tap, the input gradient, the value."""
    dtype = torch.float64 if dtype_name == "float64" else DTYPES[dtype_name]
    layout = torch.channels_last if CHANNELS_LAST else torch.contiguous_format
    net = term.to(DEVICE, dtype, memory_format=layout)
    x = ref["input"].to(DEVICE, dtype).contiguous(memory_format=layout)
    truth = ref["float64"]
    base = ref[dtype_name]

    cots = cotangents(truth["features"])
    features, grad = input_gradient(features_of, x, cots)
    leaf = x.detach().requires_grad_()
    value = net(leaf, x.flip(0))
    scale = FP16_SCALE if dtype == torch.float16 else 1.0
    (value_grad,) = torch.autograd.grad(value * scale, leaf)
    value_grad = value_grad / scale

    pairs = [
        (f"tap{i}", f, b, t)
        for i, (f, b, t) in enumerate(
            zip(features, base["features"], truth["features"])
        )
    ]
    pairs += [
        ("input_grad", grad, base["input_grad"], truth["input_grad"]),
        ("value", value, base["value"], truth["value"]),
        ("value_grad", value_grad, base["value_grad"], truth["value_grad"]),
    ]
    rows = []
    for part, candidate, baseline, reference in pairs:
        if dtype_name == "float64":
            row = {"stage": "S1", **stats(candidate.cpu(), reference)}
        else:
            row = {"stage": "S2", **compare(candidate.cpu(), baseline, reference)}
        rows.append({"net": name, "dtype": dtype_name, "part": part, **row})
    return rows


def show(row: dict) -> str:
    """One printed line of a row."""
    if row["stage"] == "S1":
        return f"rel_max {row['rel_max']:.1e} cos {row['cos']:.12f}"
    cand, base = row["candidate"], row["baseline"]
    return (
        f"{row['mark'] or '-':11s} rel_median {cand['rel_median']:.1e} / {base['rel_median']:.1e}"
        f"  cos {cand['cos']:.6f} / {base['cos']:.6f}"
    )


def main() -> None:
    """S0 per net, then S1 and S2 rows, printed and written to `OUT`."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with OUT.open("w", encoding="utf-8") as file:
        for name in KEYS:
            ref = torch.load(REFS / f"s1_{name}.pt", weights_only=False)
            term, features_of = ours(*TERMS[name])

            # S0: each of our tensors is one of upstream's, whatever the keys
            mine = set(digest(term.state_dict()).values())
            missing = mine - set(ref["weights"].values())
            buffers = {digest({k: v}).popitem()[1] for k, v in term.named_buffers()}
            print(
                name,
                "S0 weights",
                (
                    "equal"
                    if not missing - buffers
                    else f"{len(missing - buffers)} DIFFER"
                ),
                flush=True,
            )

            for dtype_name in ("float64", *DTYPES):
                for row in rows_of(name, ref, term, features_of, dtype_name):
                    file.write(json.dumps(row) + "\n")
                    if row["part"] in (
                        "tap0",
                        "input_grad",
                        "value",
                        "value_grad",
                    ) or row.get("mark"):
                        print(
                            name,
                            dtype_name,
                            f"{row['part']:10s}",
                            show(row),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
