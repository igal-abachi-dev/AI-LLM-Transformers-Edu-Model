# 50M Edu CPU smoke scorecard

This is an **engineering baseline**, not a model-quality claim. It is one FP32 optimization
step over packed, project-authored prose on a CPU-only Azure Dev Box. The planned 1-5M-token
FineWeb-Edu smoke and standard lm-eval scores remain GPU work for the home RTX machine.

## Result

- Parameters: 53,361,152
- Packed train tokens: 32
- Loss after the step: 10.397910
- Validation cross-entropy: 10.256542
- Validation perplexity: 28468.164783
- Validation bits/byte: 3.699265
- Measured CPU forward tokens/s (5 iterations): 563.869
- Allocated KV-cache bytes: 1,949,696
  (batch 1, capacity 34)
- Checkpoint logits exact after reload: True
- Harness adapter smoke: locally unit-tested; standard tasks not run

## Limitations

- One step cannot establish convergence, downstream quality, or useful generation.
- CPU throughput is not comparable to the future RTX BF16/SDPA/compile measurements.
- ARC-Easy, HellaSwag, PIQA, and optional GSM8K are configured but not downloaded here.
- Raw comparable record: `reports/runs/50m-edu-cpu-smoke.json`.
