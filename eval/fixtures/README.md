# Code/FIM fixtures v1

These small fixtures are original to MiniFrontier and licensed under Apache-2.0. They are
evaluation-only: data preparation must reject their normalized SHA-256 hashes. Execution is
opt-in and intended only for these reviewed fixtures; the scorer is not a security sandbox.

MF-054 may expand the set, but must version it rather than mutating this baseline silently.

`code_fim_v1_reference_predictions.jsonl` contains the fixture answers and exists only to prove
the scorer/report path. It is not a model result and must never be admitted to training data.

## C# fixtures (`code_csharp_fim_v1.jsonl`, 2026-09-13)

Same real-fixture/no-training-admission rules as above, one real difference: scoring is a real
`dotnet build` compile check (`score_csharp` in `src/minifrontier/evaluation/code.py`), not
Python's `ast.parse`/`compile`. Every fixture's `prompt`+`prediction`(+`suffix` for `fim`) must
reconstruct valid C# *members of a class* (a method or property body) -- scoring wraps it as
`class Fixture { ... }` and builds as a `Library` project (no entry point required), which
sidesteps two real C# rules that make bare top-level statements the wrong shape for a fixture
harness (an `Exe` project needs a real entry point; top-level statements must precede any type
declaration in the same file). No external NuGet packages are referenced, so scoring never needs
network access once the .NET SDK itself is installed -- deliberately narrower than the Python path
in one way: no functional test execution yet (`tests` is rejected on C# fixtures), a real, disclosed
scope limit, not an oversight. `dotnet`-dependent tests skip gracefully when the SDK is not on
`PATH`, matching this project's existing pattern for other optional external dependencies.

## TypeScript fixtures (`code_typescript_fim_v1.jsonl`, 2026-10-01)

Same real-fixture/no-training-admission rules as above. Added (`MF-165`) after a real
cross-language code-completion failure was observed on a live MF-070 checkpoint (TypeScript
context drifting into Java-flavored output) that the coding tier had no fixture coverage to
catch in that language at all. Scoring is a real `tsc --noEmit --strict --skipLibCheck`
type-check (`score_typescript` in `src/minifrontier/evaluation/code.py`), not Python's
`ast.parse`/`compile`. Unlike C#, no project-file or class wrapping is needed -- TypeScript
allows a standalone top-level function declaration, so every fixture's
`prompt`+`prediction`(+`suffix` for `fim`) reconstructs a plain function body directly. No
imports or `package.json` are used by any fixture, so a real check never needs network access
once the TypeScript compiler itself is installed locally -- same disclosed scope limit as C#:
no functional test execution yet (`tests` is rejected on TypeScript fixtures). `tsc`-dependent
tests skip gracefully when the compiler is not on `PATH`, matching the `dotnet` pattern above.
