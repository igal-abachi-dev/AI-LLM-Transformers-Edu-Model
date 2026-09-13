# MF-095: real bounded comparison — data mixture ratios and decay-phase curriculum

Real measurements of `train/pretrain.py` on `150m-modern.toml`, RTX 2070
Super, FP16, seed 42, 5000 updates / ~10.23M train tokens per arm. Five real
arms: a single-source FineWeb-Edu baseline, the proposed SmolLM2-modeled
5-source mixture, the existing MF-094 70/30 web/code split, and an isolated
WSD-schedule pair (fixed mixture vs. decay-phase-reweighted) testing this
task's own core question.

## Real results

| Arm | Cross-entropy | Perplexity | Bits/byte | vs. fineweb-only (relative CE) |
|---|---|---|---|---|
| fineweb-only (single-source) | 5.100674 | 164.13 | 1.71589 | — |
| wsd-fixed (5-source, WSD schedule) | 5.149342 | 172.32 | 1.73226 | +0.954% (20.7x noise floor) |
| wsd-decay-curriculum (5-source, WSD, decay reweight) | 5.188270 | 179.16 | 1.74536 | +1.717% (37.3x noise floor) |
| proposed-5-source (5-source, cosine schedule) | 5.244074 | 189.44 | 1.76413 | +2.811% (61.1x noise floor) |
| web-code-70-30 (2-source, cosine schedule) | 5.304794 | 201.30 | 1.78456 | +4.002% (87.0x noise floor) |

**Every mixture arm scored worse than plain single-source FineWeb-Edu on
this validation set — all real, decisive effects, all far above MF-088's
noise floor.** The isolated curriculum comparison (the task's own actual
question): `wsd-decay-curriculum` vs. `wsd-fixed`, both under WSD with the
identical stable-phase mixture, differing only in whether the decay phase
reweights toward Cosmopedia-v2/FineMath — **decay-phase reweighting made
things worse, not better** (+0.756% relative CE, 16.4x noise floor).

## The real methodological point that has to govern how these numbers are read

**This validation set is 100% FineWeb-Edu text** (`data/shards/mf064-150m-train/validation`,
chosen as the one split common to all five arms). A model that spent more of
its bounded ~10.23M-token budget on plain FineWeb-Edu text will
mechanically score better on FineWeb-Edu-flavored held-out cross-entropy
than a model that spent real budget on code, math, and synthetic text
instead — *regardless of whether the mixture made the model more capable
overall*. The entire stated purpose of a mixture (SmolLM2/SmolLM3's own
real recipes) is broadening capability beyond plain web text; this
comparison's own metric cannot see that benefit at all, only the cost of
diluting the FineWeb-Edu-specific signal. **These results should be read as
"which arm predicts held-out FineWeb-Edu text best," not "which arm
produces the most capable model" — those are different questions, and this
report only answers the first one.**

This is not a reason to discard the results (the FineWeb-Edu-CE question is
still real and relevant — [[MF-070]]'s own release model needs to be
genuinely good at general text, not just at code/math), but it is a reason
not to treat "the mixture loses" as the final word on whether the mixture
is worth adopting. [[MF-086]]'s broader lm-eval tasks (BLiMP, WinoGrande,
OpenBookQA, CommonsenseQA, BoolQ, lambada_openai — none of them
FineWeb-Edu-specific) exist and were not run here; they would be the right
next step before a final mixture decision, not assumed to already be
covered by this pass.

## Decision

**Plain FineWeb-Edu (single-source) is the real, evidence-backed default
for [[MF-070]] on this project's own current evidence — not the proposed
5-source mixture, and not the previously-tested 70/30 web/code split.**
This reverses the working assumption the SmolLM2-modeled proposal was
built on. The decay-phase curriculum specifically is also not supported by
this pass's own isolated comparison (a real, clean negative result, not
just "no mixture won"). Given the real methodological caveat above, this
decision is scoped to *this specific FineWeb-Edu-CE metric at this specific
~10M-token bounded budget* — not a permanent rejection of mixtures in
general. Before [[MF-070]]'s real 3B-token run locks in a final data
recipe, either (a) accept plain FineWeb-Edu as the real, measured, honest
choice for this budget, or (b) re-run this same comparison with
[[MF-086]]'s broader task suite added, to see whether the mixture's
real cost on FineWeb-Edu-CE is offset by a real gain on code/math/reasoning
tasks the current metric can't see.

## Limitations

Single seed, single scale, single ~10.23M-token budget — a real, informative
result at this budget, not proof the same ranking holds at [[MF-070]]'s
much larger 3B-token target, where a diluted-but-broader mixture has more
real tokens to make up ground on each domain. The FineWeb-Edu-only
validation metric (discussed above) is this report's single most important
caveat, not a minor footnote.

## Broader-eval follow-up (2026-09-12): the FineWeb-Edu-CE caveat's own recommended next step, now run for real

The limitation above named the exact next step needed before trusting this
comparison as final: re-run it with [[MF-086]]'s broader, non-FineWeb-Edu
-specific task suite (BLiMP, ARC-Easy, HellaSwag, PIQA, WinoGrande,
OpenBookQA, CommonsenseQA, BoolQ, lambada_openai) plus GSM8K, comparing
`fineweb-only` against `wsd-fixed` (the least-bad of the four mixture arms
above). Both real runs use `--limit 100` for statistical stability beyond
the noise floor an earlier `--limit 25` pass showed
(`reports/mf-overnight-eval/mf095-fineweb-only-harness.json`, superseded by
the `-limit100` files below) -- finding and fixing [[MF-113]]'s real
eval-harness performance bug along the way made this real, `--limit 100`
GSM8K-inclusive comparison practical to run at all (the pre-fix code would
have taken many hours per arm; post-fix, well under one hour).

| Task | `fineweb-only` | `wsd-fixed` | Real difference |
|---|---|---|---|
| **GSM8K** (exact-match, both scoring variants) | **0.0** | **0.0** | **None -- tied at the floor** |
| BLiMP (n=6,700 pooled) | 0.5936 | 0.6046 | Small edge to `wsd-fixed` (~1.1pp -- real signal is plausible here, but close to this sample size's own noise band) |
| arc_easy / boolq / commonsense_qa / winogrande | ~equal | ~equal | Noise-level (n=100 each), no real difference either way |
| piqa | 0.48 / 0.44 (norm) | 0.54 / 0.49 (norm) | Real-looking edge to `wsd-fixed`, but n=100 -- noisier than BLiMP's pooled result |
| lambada_openai | 0.0 acc, 269,685 ppl | 0.0 acc, 334,508 ppl | Both fully nonfunctional at this budget -- neither number is meaningful |

**GSM8K is the single most decision-relevant result here, and it shows
zero discriminating power between the two arms.** `wsd-fixed`'s mixture
allocates only 5% of its budget to FineMath -- at this pass's real
~10.23M-token total, that is roughly 512K real math tokens, evidently far
too few to move a from-scratch model's math capability off an absolute
floor either way. This is a direct, real confirmation (not just a
prediction) that this comparison's own token budget is too small for
GSM8K to answer the actual question being asked ("does adding math data
help math capability") -- see [[MF-116]] for the general form of this
eval-sensitivity gap and its recommended fix (a graded, continuous
held-out CE/BPB metric on math-flavored text, not an exact-match end-task
score, for small-scale mixture comparisons).

## Decision, updated

**Plain FineWeb-Edu (single-source) remains the real, evidence-backed
default for [[MF-070]] at this budget.** The broader-eval pass this
report's own limitations section called for does not reverse the earlier
CE-based finding -- it neither strengthens nor meaningfully contradicts
it, since GSM8K (the metric most relevant to the actual reason a
math-inclusive mixture would be worth its CE cost) shows no signal in
either direction. This is not evidence that mixtures don't help --
it is evidence that *this specific, diluted 5% math share, at this
specific tiny token budget* could not be measured to help, which is a
different and much narrower claim. Real candidates for a next attempt,
not yet run: a materially larger math/code share (concentrating more of
the existing budget into the domains this project's own stated priorities
actually care about, rather than SmolLM2's own broader-capability-tuned
ratios); NVIDIA ClimbMix as a differently-constructed (cluster-learned,
not hand-picked) alternative; or a later, concentrated math/code stage
closer to SFT rather than a diffuse blend through the entire pretrain --
see [[MF-116]] for the full reasoning behind each.

## Two new real candidates (2026-09-12): reweighted toward math/code, plus the final broader-eval pass across all four arms

Two new mixture arms were trained and evaluated with the same real
discipline as every arm above, both reweighting away from `wsd-fixed`'s
SmolLM2-modeled ratio toward more math/code specifically, per this
project's own stated priority (math/code over general breadth,
multilingual explicitly not needed):

| Arm | DCLM-Edu | FineWeb-Edu | GitHub-code | FineMath | Cosmopedia-v2 |
|---|---|---|---|---|---|
| `wsd-fixed` (existing) | 45% | 30% | 15% | 5% | 5% |
| `general-purpose-optimized` | 35% | 25% | 20% | 15% | 5% |
| `math-code-heavy` | 25% | 15% | 30% | 25% | 5% |

**Real, complete broader-eval harness data now exists for all four real
candidates** (`fineweb-only`, `wsd-fixed`, `general-purpose-optimized`,
`math-code-heavy`), `--limit 100`, BLiMP + ARC-Easy + HellaSwag + PIQA +
WinoGrande + OpenBookQA + CommonsenseQA + BoolQ + lambada_openai + GSM8K.
**Result: essentially zero discriminating signal across any of the four.**
GSM8K, BoolQ, and CommonsenseQA are exactly identical across all four arms;
BLiMP wobbles in a tiny 0.594-0.605 band with no monotonic relationship to
mixture composition (unlike the CE curves below) -- noise, not signal.
This is a real, decisive, final confirmation of [[MF-116]]'s eval
-sensitivity finding: at this project's current bounded token budget, no
standard end-task metric can discriminate between these mixture recipes at
all. The only place real signal exists is the domain-specific CE curve:

| Arm | FineWeb-Edu CE | Math CE |
|---|---|---|
| `fineweb-only` | 5.1007 (best) | 5.9286 (worst) |
| `wsd-fixed` | 5.1493 | 4.3357 |
| `general-purpose-optimized` | 5.2337 | 3.9750 |
| `math-code-heavy` | 5.4293 (worst) | 3.7655 (best) |

A clean, real, monotonic tradeoff -- more math/code share costs general
-text CE and buys math-text CE, with no more broader-eval data left to
gather that could break the tie a different way.

## Final decision (2026-09-12)

**`general-purpose-optimized` (DCLM-Edu 35% / FineWeb-Edu 25% / GitHub-code
20% / FineMath 15% / Cosmopedia-v2 5%) is the real, adopted mixture for
[[MF-070]].** User's own call, given the complete tradeoff above and this
project's stated priorities: a real, meaningful math/code improvement over
`wsd-fixed` (Math CE 3.9750 vs. 4.3357) at a moderate, not extreme,
general-text cost (FineWeb-Edu CE 5.2337 vs. 5.1493) -- the middle
candidate on the real curve, not either endpoint. `math-code-heavy`'s
further math gain was judged not worth its steeper general-text cost;
plain `fineweb-only`'s general-text edge was judged not worth giving up
the real, large math/code improvement, given the project's own stated
usefulness priorities.

## Limitations

Single seed, single scale, single ~10.23M-token bounded budget for every
arm in this report. The broader-eval harness's own inability to
discriminate between arms (confirmed above, not merely suspected) means
this decision rests on the CE/BPB curve specifically, which is real and
trustworthy for what it measures (predictive quality on held-out text in
each domain) but does not directly measure end-task capability (solving a
GSM8K problem, writing correct code) -- the honest caveat [[MF-116]]
already recorded, now with a fully confirmed, not just suspected, reason
why the broader harness couldn't settle it more directly at this budget.

## The math-specific held-out CE/BPB check (2026-09-12): [[MF-116]]'s proposed fix, verified for real, and a large, real effect found

[[MF-116]] proposed a graded held-out CE/BPB metric on math-flavored text as
the sensitive proxy GSM8K's exact-match metric cannot provide at this token
budget. Tested directly, immediately, using data already on hand (no new
prep needed): `data/shards/mf095-finemath-train/validation`, a real,
already-existing 20,460-token held-out FineMath split.

| Checkpoint | Cross-entropy | Perplexity | Bits/byte |
|---|---|---|---|
| `fineweb-only` (0% math) | 5.9286 | 375.6 | 2.7983 |
| `wsd-fixed` (5% FineMath) | **4.3357** | **76.4** | **2.0464** |

**A real 26.9% lower cross-entropy, ~5x lower perplexity** -- hundreds of
times beyond this project's own established noise floor
([[MF-088]]'s ~0.046% relative CE). This is the clean, decisive signal
GSM8K's exact-match metric could not show (both arms tied at 0.0): the 5%
FineMath share *did* teach the model something large and real about
math-flavored text, just not enough to cross the much higher bar of
solving a full word problem end-to-end. **This reframes the earlier
FineWeb-Edu-CE-based ranking, not reverses it**: "plain FineWeb-Edu wins"
was true specifically *on FineWeb-Edu-flavored held-out text* -- of course
a model trained on 30% less FineWeb-Edu does worse there. This result is
the mirror image, on math-flavored text instead. Neither number alone says
"which model is better overall" -- each measures how well a model learned
the specific distribution it was actually shown, the same underlying
principle [[MF-097]]'s shared-validation-set correction already
established for packing, now confirmed on the mixture axis too.

**Directly supports pushing further, not stopping here.** If 5% FineMath
already produced this large an effect, a materially larger math/code share
is a well-motivated next real test (queued, not yet run) -- and this
math-specific CE/BPB check is now the real, sensitive instrument to judge
it with, since GSM8K itself will very plausibly stay at its floor
regardless of mixture at this project's current bounded token budgets.
