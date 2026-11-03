# Measurements

A chain of truth. Each stage proves one step against the level below, which it reads from a file
recorded once; no stage reruns a lower one. Numbers here are read by a person. Nothing in this
folder runs on every change: run a stage when the code it covers changes.

Recorded files go to `measure/refs/`, which is not in git.

## Correctness

| Stage | Script | Proves | Against |
| --- | --- | --- | --- |
| S0 to S2 record | `s1_record.py` | upstream at fp64 and each dtype | needs sensecraft 0.3.13 installed, never `src/` |
| S0 | `s2_nets.py` | every converted weight equals upstream's | upstream weights |
| S1 | `s2_nets.py` | each perceptual term's features, input gradient, value and value gradient at fp64 | upstream at fp64 |
| S2 | `s2_nets.py` | the same at fp32, bf16 and fp16 | fp64 reference, upstream at that dtype as baseline |
| S5 record | `s5_record.py` | the 35 fixed losses at fp64, fp32 and bf16 | needs sensecraft 0.3.13 installed |
| S5 | `s5_fixed.py` | value and input gradient of every fixed loss | the S5 record |
| S6 | `s6_compose.py` | ranges, weights, monitors, shift, video frames, `value_and_seed` | stub terms with known outputs |
| S7 | `s7_near.py` | each perceptual term's gradient when the prediction is near the target, in fp16, bf16, TF32 and strict fp32 | fp64; draws `figures/s7_near.svg` from every recorded label |
| S8 | `s8_shift.py` | how a random shift spreads a perceptual term's gradient: persistence over calls, peak and high-frequency share | the unshifted term |

Rows report the max, mean and median of the relative and absolute error and the cosine, for the
candidate and the baseline. `clear issue` marks a row at 10x the baseline, `large diff` at 2x. A mark
sends a person to look; a mark carried only by `rel_max` on near-zero reference values is not a
kernel error.

## Speed

| Level | Script | Reads |
| --- | --- | --- |
| T1 | `t1_floor.py` | the card's conv, matmul, attention and copy rates, one file per card |
| T2 | `t2_layers.py` | every layer of a term alone, forward and input backward, as a ratio to T1 |
| T3 to T5 | `t3_terms.py` | the net, the distance and the whole term, per dtype, layout and compile scope |

Each level owns the ratio it adds to the one below, so a slow term is named without a long run.

## Running

Stages that touch DINOv3 need `HF_HOME` pointing at a cache holding the gated checkpoints, or a
stand-in with random weights; point `SENSECRAFT_CACHE` at a separate folder for stand-ins so their
converted weights never reach the real cache. Every script runs with plain `python` and no
arguments; its knobs are the UPPER_CASE globals at its top.

## Tried and not kept

Measured at batch 32, 256 px, bf16, channels-last, compiled per term, on one RTX 5090.

| Change | Result | Why not kept |
| --- | --- | --- |
| DINOv3 q, k and v as one projection | 21.5 to 31.1 ms compiled | slower under compile |
| Target forward on a second stream | 0.4 ms better than the same split graph, 1.5 ms worse than one compiled term | the split costs more fusion than the overlap gives |
| Triton VGG block: pool code, routing, tap and ReLU masks inside the input-gradient kernel | 23.10 to 23.35 ms device time | the four-operand load costs what the removed passes saved |
| Plain Triton 3x3 conv with bias and ReLU | ties cuDNN at 64 channels, 8 to 10% slower at 128 and 256 | no gain over cuDNN's fused op |
