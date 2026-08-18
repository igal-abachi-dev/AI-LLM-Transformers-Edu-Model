# MiniFrontier task system

This directory is the execution source of truth derived from [`plan.md`](../plan.md).

- [`backlog.md`](backlog.md) is the human-readable, dependency-ordered backlog.
- [`jira-issues.xml`](jira-issues.xml) is a Jira issue-export-style XML representation of the same work items.

## Workflow

1. Select the first `Ready` task whose dependencies are `Done`.
2. Change its state to `In progress` before editing implementation files.
3. Implement only that task and its stated acceptance criteria.
4. Run the narrow tests first, then the full fast test suite.
5. Record commands, results, and important decisions in the task notes.
6. Mark it `Done` only when every acceptance criterion passes.

Allowed states are `Planned`, `Ready`, `In progress`, `Blocked`, and `Done`.

## Jira XML note

The XML uses Jira's issue-export/RSS field vocabulary so it is readable by Jira-oriented tooling and easy to transform. Jira Cloud's supported generic bulk import paths are CSV and JSON rather than an arbitrary issue XML file, so check the target Jira deployment before importing. The stable `MF-NNN` IDs, components, labels, dependencies, and acceptance criteria are preserved in the XML for lossless conversion.

## Completion gates

- **M0–M3:** a correct Edu model can train, save, reload, and generate with KV-cache parity.
- **Evaluation gate:** baseline metrics exist before architectural experiments.
- **M4–M8:** Modern attention, performance, FIM/code, Muon, and SFT are implemented and measured.
- **M9 / V1 complete:** matched 150M Edu and Modern artifacts are published with reproducible evaluations.
- **M10:** MF-069 may complete software/preflight work before the scheduled RTX session; MF-070 owns
  the real 350M/500M measurements after V1 and cannot be replaced by estimates.

The user has scheduled CUDA execution after M10 implementation. This changes execution order, not
the definition of done: GPU-dependent M5/M9 tasks and actual 150M artifacts remain open until the
post-M10 hardware run. MF-069 exists specifically so M10 implementation/preflight can close honestly;
MF-070 and every CUDA acceptance criterion stay open until measured on the target GPU.
