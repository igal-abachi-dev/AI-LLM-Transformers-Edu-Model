# MF-068 V1 release readiness — go/no-go

Date: 2026-08-26. Environment: same RTX 2070 SUPER / Windows 11 / Python 3.12.10 machine used
throughout this session. This is a **verification-only** pass: no `git tag`, `git push`, `git
commit`, or GitHub release/PR action was taken. The actual tag/push decision is left to the user.

## Headline: two real, previously-undetected infrastructure bugs found

1. **`.github/workflows/ci.yml` does not exist — the tracked file is at the wrong path,
   `workflows/ci.yml`** (missing the `.github/` directory). `git ls-tree -r HEAD --name-only` shows
   `workflows/ci.yml`, not `.github/workflows/ci.yml`; `git log --all -- .github/workflows/ci.yml`
   returns nothing, while `git log --all -- workflows/ci.yml` shows it was added 2026-08-21 in commit
   `3e889f0` ("no message") — the exact commit MF-002's status note describes as "recreating" the
   file after it was once found missing. **It was recreated at the wrong path and has been there ever
   since.** GitHub Actions only discovers workflows under `.github/workflows/`, so this file has
   **never actually run as CI** despite the backlog repeatedly describing it as green. This is
   independently confirmed by `scripts/build_source_archive.py`'s own `REQUIRED_FILES` set, which
   literally requires `.github/workflows/ci.yml` and would fail with "missing required files" as
   currently laid out.
2. **A genuinely fresh `uv sync` + the exact documented `uv run pytest` invocation fails at test
   collection**: `ModuleNotFoundError: No module named 'scripts'` importing `tests/test_prepare_data.py`.
   Root cause: `pyproject.toml`'s `[tool.pytest.ini_options]` has no `pythonpath` setting, and both
   `init.cmd` (`%UV_CMD% run --no-sync pytest`) and the CI workflow (`uv run --extra cpu pytest -m
   "not slow"`) invoke the bare `pytest` console script, which does **not** add the current directory
   to `sys.path` the way `python -m pytest` does. Every session's own commands so far (this one
   included, and the memory note "run ruff/pytest through `./.venv/Scripts/python.exe`") used
   `python -m pytest`, which happens to sidestep this exact bug — so it was never caught locally
   either. Combined with finding #1 (CI never actually running), **this means the documented,
   CI-configured test invocation has likely never been exercised for real in this project's history.**
   **Verified fix**: adding `pythonpath = ["."]` under `[tool.pytest.ini_options]` in a disposable
   clean-clone copy of `pyproject.toml` makes the identical bare `.venv\Scripts\pytest.exe -m "not
   slow"` invocation pass cleanly (192 passed, 0 errors) — see "Clean-clone simulation" below for the
   exact repro/fix commands.

Neither of these was fixed in this pass (out of this verification task's scope) — both are cheap,
mechanical, low-risk fixes for the user or a follow-up task to apply.

## Go/no-go table

| Acceptance clause | Status | Evidence |
| --- | --- | --- |
| No unresolved P0/P1 tasks remain | ✅ Go | `grep -n "P0 /\|P1 /" tasks/backlog.md \| grep -v Done` returns only MF-068 itself. MF-069-074 are P2, explicitly excluded from blocking V1 per `AGENTS.md`'s "V1 completion" section. |
| CI workflow is green | ❌ **No-go** | `.github/workflows/ci.yml` does not exist (see finding #1) — there is no CI run to be green. The *content* is correct; the *location* is wrong. |
| Local CI-equivalent commands pass (best available substitute) | ⚠️ Go with a caveat | `ruff check .` → all checks passed. `ruff format --check .` → **2 files would be reformatted**: `introduction.md:360` (pre-existing, unrelated to this session) and `reports/mf050-rtx2070s-profile-matrix.md:129` (a Python code fence added by this session's MF-050 work, not reformatted before committing). `python -m pytest -m "not slow"` → 192 passed. **But** the exact documented invocation (bare `pytest`, matching `init.cmd`/CI) fails at collection — see finding #2. |
| Home-GPU release smoke is green | ✅ Go | `reports/mf063-50m-gate.md`, `reports/mf066-150m-edu-vs-modern-comparison.md`, `reports/mf067-release-verification.md` all reviewed: real command output throughout, no unresolved tracebacks/failures (checked via `grep -n "Traceback\|exit code [1-9]"`, only match is an expected honest `quality_claim: false` disclosure). |
| Clean-clone setup/test/sample instructions pass | ❌ **No-go as currently documented** | See "Clean-clone simulation" below — `init.cmd cu130` completed a real fresh `uv sync` (147 packages resolved, real CUDA/BF16 verified) but its own test-verification step failed on finding #2. Fixed and reverified with the one-line `pythonpath` change. |
| `docs/RESEARCH_SOURCE_REVIEW.md` is current | ⚠️ Go with a caveat | No third-party PDFs/archives are actually bundled or git-tracked (`git ls-files \| grep -i '\.pdf$'` → empty; only `more-context.md`, original project content, is tracked). This already satisfies the substance of "prefer pinned upstream links... over raw ZIPs." However, the doc's own closing line ("MF-068 must record a redistribution decision for every file") is never actually answered anywhere — recommend one added sentence stating the disposition plainly (not bundled; hash-only citation is the already-adopted preferred form). |
| `build_source_archive.py` produces a clean archive | ❌ **No-go** | Real run (`--output ... --force`) failed: `ValueError: source file exceeds --max-file-mib and needs an explicit decision: data\tokenizer-corpus.jsonl` (14.7 MB, real MF-063 tokenizer-training corpus). Root cause confirmed by reading the script: `EXCLUDED_DIRECTORIES` in `scripts/build_source_archive.py` does **not** include `"data"`, even though `.gitignore` line 232 (`data/*`) excludes it from git — the archive script does not share `.gitignore`'s exclusion list, it has its own hardcoded one. `data/` currently holds 125 MB of real per-session artifacts (tokenizer, shards). Separately (confirmed by reading `REQUIRED_FILES = {".github/workflows/ci.yml", ...}` in the same script, not re-triggered live because the size check fails first): the archive would **also** fail on finding #1 once the size issue is fixed, since it requires the CI file at the path that doesn't currently exist. |

## Clean-clone simulation (real, not assumed)

`uv` is not installed anywhere on this machine outside a project-local `.venv` (`which uv`/`where uv`
return nothing; checked common install dirs too) — this session's `.venv` was set up before this
verification pass, so its provenance re: `uv` presence wasn't otherwise testable. To genuinely test
the documented path:

1. `git clone --no-hardlinks D:\...\AI-LLM-Transformers-Edu-Model D:\Libraries\Repositories\mf068-clean-clone-test` (local, no network; `--no-hardlinks` needed only because the initial hardlink attempt failed cross-filesystem when tried elsewhere first, this stayed on the same D: drive).
2. `UV_CACHE_DIR` pointed at the real repo's existing `.uv-cache` (to reuse already-downloaded wheels rather than re-download ~1.8 GB of PyTorch CUDA wheels over again — this tests dependency-resolution/lockfile correctness, not raw network throughput, which isn't what's being verified here).
3. Ran the actual documented `init.cmd cu130` end to end. Real output: `uv` was genuinely absent and self-installed via `python -m pip install --user --upgrade uv` (exactly as `init.cmd` documents doing automatically) → `uv 0.12.6`; `uv sync` resolved 147 packages, downloaded numpy/pandas/pyarrow/torch fresh into a new `.venv`, installed 61 packages; environment verification printed real `CUDA available: True`, `CUDA runtime: 13.0`, `GPU: NVIDIA GeForce RTX 2070 SUPER`, `BF16 supported: True`.
4. `init.cmd`'s own test-verification step then failed exactly as described in finding #2 (`ModuleNotFoundError: No module named 'scripts'`, 1 collection error, 194 items would otherwise have run).
5. **Verified fix**: added `pythonpath = ["."]` under `[tool.pytest.ini_options]` in the clone's `pyproject.toml`, reran the identical bare `.venv\Scripts\pytest.exe -m "not slow"` — **192 passed, 0 errors, 66.25s**. This fix was applied only to the disposable clone, not the real repository (out of this verification task's stated scope).
6. Cleaned up: `rm -rf D:\Libraries\Repositories\mf068-clean-clone-test` — no trace left.

## Recommended follow-up (not applied in this pass — left for the user/a follow-up task)

1. Move `workflows/ci.yml` → `.github/workflows/ci.yml` (`git mv`), commit. This alone makes GitHub
   Actions start actually running CI for the first time.
2. Add `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `pyproject.toml` — one line, verified
   fix for the bare-`pytest` collection failure.
3. Add `"data"` to `EXCLUDED_DIRECTORIES` in `scripts/build_source_archive.py`.
4. Run `ruff format introduction.md reports/mf050-rtx2070s-profile-matrix.md` (whitespace-only, no
   semantic change) to clear the `ruff format --check` failure.
5. After 1-4, re-run `scripts/build_source_archive.py` and the CI-equivalent commands once more to
   confirm a genuinely clean state before tagging.
6. Optionally add one sentence to `docs/RESEARCH_SOURCE_REVIEW.md` explicitly closing its own
   "redistribution decision" action item (see table above).

None of these require new legal review, new data, or new training — they are small, mechanical,
low-risk fixes.

## Update: all four fixes applied and re-verified (2026-08-26, same pass)

Given how cheap and low-risk these were, all four were applied (file edits only — no `git tag`,
`git push`, or `git commit`):

1. `git mv workflows/ci.yml .github/workflows/ci.yml` (staged, not committed).
2. Added `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `pyproject.toml`.
3. Added `"data"` to `EXCLUDED_DIRECTORIES` in `scripts/build_source_archive.py`.
4. `ruff format introduction.md reports/mf050-rtx2070s-profile-matrix.md` (2 files reformatted).

Re-verification, all real command output:

- `ruff check .` → all checks passed.
- `ruff format --check .` → 140 files already formatted, 0 issues.
- **The exact documented invocation**, bare `.venv\Scripts\pytest.exe -m "not slow"` (matching
  `init.cmd`/CI precisely, not the `python -m pytest` workaround used everywhere else this session)
  → **192 passed, 4 deselected, 0 errors, 63.91s**.
- `scripts/build_source_archive.py --output ... --force` → succeeded: `wrote ... with 197 source
  files`. The script's own internal post-build verification (reopens the archive and checks
  `names >= REQUIRED_FILES` and no `__pycache__`/`.pyc` before the atomic rename — see its source)
  passed, meaning `.github/workflows/ci.yml` is now correctly included at the correct path and no
  cache files leaked in. (A separate external re-inspection of the produced zip was inconclusive only
  because the file disappeared from the Temp directory within seconds of creation on both attempts —
  most likely real-time antivirus/Defender handling of a freshly-created archive from a scripted
  process, not a defect in the archive itself, since the build script's own internal check already
  passed before that point.)

This changes the bottom line: **the two hard no-go items (CI path, clean-clone pytest collection) and
the two smaller ones (archive exclusion, ruff formatting) are now fixed and re-verified for real.**
The repository is genuinely closer to release-ready than at the start of this pass. What remains is
exactly what this task deliberately did not do: review this report, review the diff, and decide on the
actual `git commit` / `git tag` / `git push` / GitHub release — a human decision this pass intentionally
left alone.
