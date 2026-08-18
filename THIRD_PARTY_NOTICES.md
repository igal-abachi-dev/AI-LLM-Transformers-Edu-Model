# Third-party notices

MiniFrontier is licensed under Apache-2.0. Runtime and development dependencies keep their own licenses; `uv.lock` records the resolved package versions.

The repository contains research papers and reference material under `docs/`. Their presence does not relicense them under Apache-2.0. Before a public repository release, confirm that every locally stored paper, archive, and other reference may be redistributed; otherwise replace it with a citation or download instruction.

The audited titles, review hashes, project relevance, and publication disposition are recorded in
`docs/RESEARCH_SOURCE_REVIEW.md`. A review hash proves which local copy informed a decision; it does
not grant redistribution rights.

Training data and trained weights are separate artifacts. They must ship with their own source manifests, license analysis, hashes, and model cards under the rules in `docs/DATA_GOVERNANCE.md`.

No third-party model implementation is copied into the MiniFrontier neural core. Optional `transformers` usage is limited to development-time primitive parity checks and interoperability experiments.
