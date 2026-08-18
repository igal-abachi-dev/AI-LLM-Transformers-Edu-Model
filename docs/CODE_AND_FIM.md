# Code data and FIM experiments

The code path is deliberately manifest-first:

```text
provenance/license validation -> sensitive/generated/vendor filters
-> disk exact/near dedup + evaluation exclusion -> approved manifest
-> deterministic PSM FIM mixture -> immutable token shards -> matched training -> scoring
```

`scripts/prepare_code.py` rejects records without approved provenance and emits only aggregate
rejection counts. `scripts/apply_fim.py` defaults to the frozen 15% experiment rate and records the
seed and `fim-psm-v1` transform. Transformed records preserve the original content identity for
stable train/validation splitting.

`scripts/compare_fim.py` starts both arms from identical weights, checks equal target-token budgets,
retains manifest hashes and resumable checkpoints, and labels bounded CPU output as engineering
evidence. `scripts/eval_code.py` reports FIM exact match, syntax, compile, trusted-fixture functional
results, exact/near contamination, and optional matched general-language metrics. Model-quality or
effect-size claims require repeated home-GPU runs and the MF-063 gate.
