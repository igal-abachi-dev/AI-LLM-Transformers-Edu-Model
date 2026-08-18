# Code/FIM fixtures v1

These small fixtures are original to MiniFrontier and licensed under Apache-2.0. They are
evaluation-only: data preparation must reject their normalized SHA-256 hashes. Execution is
opt-in and intended only for these reviewed fixtures; the scorer is not a security sandbox.

MF-054 may expand the set, but must version it rather than mutating this baseline silently.

`code_fim_v1_reference_predictions.jsonl` contains the fixture answers and exists only to prove
the scorer/report path. It is not a model result and must never be admitted to training data.
