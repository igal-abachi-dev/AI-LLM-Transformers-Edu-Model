"""Frozen byte-level BPE contract and deterministic training/loading helpers.

Beginner's map of this file
---------------------------
A model cannot read letters. Text is first chopped into **tokens** -- chunks of
characters that sit somewhere between single letters and whole words -- and each
token has an ID number. From then on the model only ever sees lists of integers.

Why chunks rather than words? There are millions of words, plus typos, plus code,
plus every other language, plus emoji. Byte-level BPE ("byte pair encoding")
solves that by starting from raw bytes and repeatedly merging the most frequent
adjacent pair into a new token. Common words like " the" end up as one token;
something unusual falls back to a few pieces; and because every byte is in the
alphabet, *nothing is ever unrepresentable* -- there is no "unknown token".

This project freezes one 16,384-token vocabulary for every model size, so a
tokenizer trained once can be compared across experiments. Briefly raised to
32,768 (2026-09-06) on Tao et al.'s (arXiv:2407.13623) vocabulary-scaling
theory, then reverted (2026-09-08, see `docs/IMPLEMENTATION_DECISIONS.md`)
after real matched-wall-clock training evidence: a bounded comparison found a
reproducible ~1% BPB regression at 32k across three independent retrains, a
periodic-validation curve showing the gap *widening* rather than closing
across the training budget (the opposite of what a short-budget-recovery
explanation predicts), and a fertility check ruling out tokenizer inefficiency
as the cause (32k was actually *more* fertility-efficient, not less). Digit-
splitting pre-tokenization (Llama 3/Qwen-style, groups of at most 3 -- see
``train_byte_bpe``'s ``digit_split`` parameter) was evaluated alongside the
32k vocabulary but never adopted: it measured as a real fertility *cost*, and
its actual purpose (arithmetic capability) was never evaluated at all, so
turning it on for the 16k vocabulary now would be an untested combination.
The parameter remains available for a future, properly isolated test. The
first fourteen IDs are reserved for markers that never appear in ordinary
text::

    <|pad|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>
    <|fim_prefix|> <|fim_suffix|> <|fim_middle|> <|tool_call|> <|tool_result|>
    <|eot|> <|file_sep|> <|repo_name|>

Those markers are the entire mechanism behind chat roles. There is no "assistant
mode" inside the model -- just a token that means the assistant's turn starts
here. It is also why prompt injection is possible at all: text that smuggles in
convincing markers can be read as structure rather than content.

``<|eot|>`` (MF-103, 2026-09-09) is the chat/SFT *turn* boundary -- distinct
from ``<|eos|>``, which keeps its original, sole role marking *document*
boundaries in the pretraining pack (``data.py``). Before this split, one
token meant both "this document is finished" and "my conversational turn is
finished," which a base-pretrained-then-SFT'd model was being asked to learn
as two different behaviors from one symbol -- the same problem real
production tokenizers avoid (Llama 3's ``<|end_of_text|>`` vs ``<|eot_id|>``;
Qwen's ``<|endoftext|>`` vs ``<|im_end|>``). ``<|file_sep|>``/``<|repo_name|>``
(MF-104, reserved the same pass, following StarCoder2/Qwen2.5-Coder's real
convention) mark file and repository boundaries for a future repo-level
(multi-file) packing task -- reserved now because doing so later would move
every ID after them, but unused until that task exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence, Split
from tokenizers.trainers import BpeTrainer

# Frozen for the whole project. Commercial models use 100k-200k; the idea is the
# same, and a smaller vocabulary keeps the tied embedding table affordable here.
VOCAB_SIZE: Final = 16_384
# Isolate runs of at most 3 digits before the ordinary byte-level pass, so
# "123456" pre-tokenizes as "123" + "456" rather than merging into one token
# the way GPT-2's regex would. `Sequence` runs this first; ByteLevel then
# further splits whatever's left exactly as before, so this changes nothing
# about non-digit text. Real MF-090 evidence (see this module's docstring)
# found only a fertility *cost* from this, never tested for a trained-quality
# benefit, so it stays off by default -- `train_byte_bpe` still accepts
# `digit_split` for a future properly isolated test, rather than hardcoding
# this away entirely.
_DIGIT_SPLIT_PATTERNS: Final[dict[str, Regex]] = {
    "no_leading_space": Regex(r"\d{1,3}"),
    "leading_space": Regex(r" ?\d{1,3}"),
}
# "individual": every digit isolated one at a time (MF-100's research pass:
# confirmed independently across StarCoder2, Llama/Llama-2, Mistral Tekken,
# and SmolLM2's own real tokenizers -- a broad, repeated industry convention,
# not an isolated choice). Uses the stock `Digits(individual_digits=True)`
# pre-tokenizer class, not a regex `Split` like the two variants above, so it
# is handled specially in `train_byte_bpe`'s pipeline construction rather
# than living in `_DIGIT_SPLIT_PATTERNS`. A genuinely different granularity
# from the two variants above (which group up to 3 digits) -- never tested
# in this project in any form.
_DIGIT_SPLIT_MODES: Final[frozenset[str]] = frozenset(_DIGIT_SPLIT_PATTERNS) | {"individual"}
DIGIT_SPLIT_MODE: Final = "none"
# GPT-4/cl100k_base's real pre-tokenization regex (MF-100), verified directly
# against tiktoken's own primary source (openai/tiktoken's
# tiktoken_ext/openai_public.py), not assumed from a secondary summary --
# an earlier WebSearch summary of this exact pattern misquoted the digit
# clause as `\p{N}{2,}`, caught only by checking the primary source. Ported
# from PCRE's possessive quantifiers (`?+`/`++`/`{1,3}+`) to the plain greedy
# equivalents (`?`/`+`/`{1,3}`), since the `tokenizers` library's own `Regex`
# silently mis-parses the possessive suffix as an unrelated repeated-group
# quantifier rather than rejecting it outright -- verified directly: with the
# possessive form, "123456789" pre-tokenized as one unsplit piece instead of
# capping at 3 digits. Possessive-vs-greedy only changes backtracking
# performance on pathological input, not the match itself, for well-formed
# text. Real, richer than this project's own GPT-2-style default: caps digit
# runs at 1-3 (like `digit_split`, but built into the split itself), handles
# contractions case-insensitively, and separates letter/number/punctuation/
# whitespace runs more finely.
_GPT4_REGEX_PATTERN: Final[Regex] = Regex(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s"""
)
#(adapted from PCRE possessive quantifiers to standard greedy quantifiers so Hugging Face’s tokenizers engine can parse it properly). ,(words, numbers ≤ 3 digits, punctuation, whitespace)
#It matches text using 8 branches separated by | (alternation)
#Matches an apostrophe ' followed by common English contraction endings case insensitive: 's, 'd, 'm, 't, 'll, 've, or 're.
#Matches 1 or more Unicode letters (across all human languages) with optional leading space
#Matches any Unicode number character  at least 1 and at most 3 digits.
#One or more characters that are not whitespace, letters, or numbers (i.e., punctuation, symbols, math operators, emojis) followed by optional newline
#partition all forms of whitespace cleanly
#GPT-2 Regex Limitations vs. GPT-4:
#Uncapped digits (\p{N}+): GPT-2 lets long numbers like "123456789" stay together as a single pre-token, causing BPE to merge arbitrary multi-digit numbers and hurting arithmetic reasoning. GPT-4 fixes this by capping digit runs to \p{N}{1,3}.
#Case-sensitive contractions: GPT-2 only matches lowercase contractions ('s, 'm, 'll), missing uppercase contractions like 'S or 'LL. GPT-4 uses (?i:...).
#Leading characters: GPT-2 only allows an ASCII space ? before letters/symbols, whereas GPT-4 uses [^\r\n\p{L}\p{N}]? to prevent newlines from accidentally binding to the start of words.

#The main pathology of legacy tokenizers (like GPT-2) was unbounded digit runs (\p{N}+), which merged arbitrary phone numbers, timestamps, and IDs into monolithic tokens. 
#Because _GPT4_REGEX_PATTERN already includes \p{N}{1,3}, it isolates numbers into small, predictable chunks at the root stage. A second split stage was redundant at best.




# GPT-4o/o200k_base's real pre-tokenization regex (MF-100's wider tokenizer
# survey), verified directly against tiktoken's own primary source
# (openai/tiktoken's tiktoken_ext/openai_public.py), already in plain greedy
# form (no possessive-quantifier porting needed, unlike cl100k_base above).
# Genuinely different from cl100k_base, not just a bigger vocab on the same
# pattern: explicit case-transition splitting via two alternatives (an
# uppercase/titlecase run followed by lowercase, and vice versa) instead of
# cl100k's single undifferentiated `\p{L}+` clause -- a real, more refined
# rule for camelCase/PascalCase-style word boundaries, confirmed empirically
# ("camelCaseWord" -> "camel"+"Case"+"Word", "PascalCase" -> "Pascal"+"Case").
# Digit cap stays `\p{N}{1,3}`, same as cl100k.
_O200K_REGEX_PATTERN: Final[Regex] = Regex(
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
# this has Vocabulary Fragmentation at Smaller Scales , so gpt4 is better for smaller models, (CamelCase splits, word-attached contractions)


PRETOKENIZER_MODE: Final = "gpt2"
# MF-100 (2026-09-12): gpt4-style regex is the real, adopted default for Modern
# specifically (`reports/mf100-stage2-comparison.md` -- a real, corroborated
# fertility-and-BPB win), Edu keeps the plain default above unaffected. Two
# named constants, not a preset parameter on `train_byte_bpe` itself: this
# low-level function has no preset awareness by design, the same followed
# convention `TrainingConfig` already uses for WSD/cautious-weight-decay
# (`AGENTS.md`) -- preset selection belongs in a caller (`scripts/
# train_tokenizer.py`'s `--preset` flag), not baked into the trainer.
EDU_PRETOKENIZER_MODE: Final = "gpt2"
MODERN_PRETOKENIZER_MODE: Final = "gpt4"
# Bumped 2026-09-09 (MF-103/MF-104): three tokens (<|eot|>, <|file_sep|>,
# <|repo_name|>) appended after the original eleven. Purely additive -- IDs
# 0-10 are unchanged -- but this is real metadata for anyone inspecting a
# tokenizer_config.json later, so the version number reflects it.
TOKENIZER_VERSION: Final = 2
SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|eot|>",
    "<|file_sep|>",
    "<|repo_name|>",
)
# IDs 0..13, assigned by position: <|pad|> is 0, <|bos|> is 1, and so on. These
# are part of the frozen contract, because a checkpoint trained with <|eos|> as 2
# produces nonsense if reloaded against a tokenizer that numbered them otherwise.
# New tokens are only ever appended (never inserted) for exactly this reason --
# 0-10 were already frozen before <|eot|>/<|file_sep|>/<|repo_name|> (11-13)
# were added (MF-103/MF-104), so no existing ID moved.
SPECIAL_TOKEN_IDS: Final[dict[str, int]] = {
    token: index for index, token in enumerate(SPECIAL_TOKENS)
}


@dataclass(frozen=True, slots=True)
class TokenizerContract:
    version: int = TOKENIZER_VERSION
    vocab_size: int = VOCAB_SIZE
    add_prefix_space: bool = False

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return SPECIAL_TOKENS

    @property
    def special_token_ids(self) -> dict[str, int]:
        return SPECIAL_TOKEN_IDS.copy()


class MiniFrontierTokenizer:
    """Thin wrapper that validates the immutable MiniFrontier token contract.

    The heavy lifting is done by the ``tokenizers`` library; this class exists to
    guarantee the parts the model depends on, above all that the special tokens
    kept their exact IDs.
    """

    def __init__(self, tokenizer: Tokenizer, contract: TokenizerContract | None = None) -> None:
        self.backend = tokenizer
        self.contract = contract or TokenizerContract()
        self._validate_special_tokens()

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|eos|>"]

    @property
    def eot_id(self) -> int:
        """The chat/SFT turn-boundary marker (MF-103) -- never used for document boundaries."""

        return SPECIAL_TOKEN_IDS["<|eot|>"]

    def _validate_special_tokens(self) -> None:
        """Refuse a tokenizer whose marker IDs drifted -- a silent, ruinous mismatch.

        A token entirely *absent* from this tokenizer (``actual_id is None``)
        is tolerated, not rejected: it means this tokenizer predates that
        token's introduction (e.g. an already-published release's tokenizer,
        trained before MF-103 added ``<|eot|>``/``<|file_sep|>``/
        ``<|repo_name|>``) -- the model trained against it never learned
        anything about that token either, so there is no drift to catch, and
        rejecting it would make every already-published checkpoint's own
        tokenizer permanently unloadable the moment `SPECIAL_TOKENS` grows.
        A token that *is* present at the wrong ID is always rejected -- that
        is real drift, never tolerated regardless of when the tokenizer was
        trained.
        """

        for token, expected_id in SPECIAL_TOKEN_IDS.items():
            actual_id = self.backend.token_to_id(token)
            if actual_id is None:
                continue
            if actual_id != expected_id:
                raise ValueError(
                    f"special token {token!r} must have ID {expected_id}, got {actual_id}"
                )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # add_special_tokens=False keeps the library from inserting markers of its
        # own: in this project the caller decides explicitly, via the flags below.
        # Not doing so silently changes what the model is trained to expect.
        token_ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self.backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def save(self, directory: str | Path, *, model_max_length: int = 2_048) -> None:
        if model_max_length <= 0:
            raise ValueError("model_max_length must be positive")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        tokenizer_path = target / "tokenizer.json"
        self.backend.save(str(tokenizer_path))
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        config = {
            "tokenizer_version": self.contract.version,
            "requested_vocab_size": self.contract.vocab_size,
            "actual_vocab_size": self.vocab_size,
            "model_max_length": model_max_length,
            "add_prefix_space": self.contract.add_prefix_space,
            "special_tokens": self.contract.special_token_ids,
            "tokenizer_sha256": digest,
        }
        (target / "tokenizer_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> MiniFrontierTokenizer:
        root = Path(directory)
        backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        instance = cls(backend)
        config_path = root / "tokenizer_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_hash = config.get("tokenizer_sha256")
            actual_hash = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError("tokenizer.json hash does not match tokenizer_config.json")
        return instance


#A pre-tokenizer acts as a hard boundary maker:
#It splits raw text into a sequence of isolated chunks (matches).
#BPE merges are strictly forbidden from crossing chunk boundaries.
#This guarantees that:
#Letters don't merge with numbers.
#Words don't merge with punctuation or newlines.
#Contractions (e.g., 's, 'll) are isolated.
#Numbers are capped at 1–3 digits to facilitate mathematical and digit-level reasoning.
def train_byte_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
    digit_split: str = DIGIT_SPLIT_MODE,
    pretokenizer: str = PRETOKENIZER_MODE,
) -> MiniFrontierTokenizer:
    """Train deterministic byte-BPE from an already deterministic text stream.

    "Training" a tokenizer is nothing like training the model -- there is no
    gradient here. It just counts which adjacent pairs of symbols occur most often
    in the corpus and merges them, over and over, until the vocabulary is full.

    The order of ``texts`` affects the merge counts, so the caller is responsible
    for handing over a stream that is already deterministic; otherwise two "same"
    tokenizers would disagree about token IDs.

    ``digit_split`` selects the pre-tokenizer's digit-isolation rule: the frozen
    default ``"none"`` (no digit isolation at all, GPT-2-style long digit runs
    merge freely -- what every real checkpoint this project has trained uses),
    two evaluated-but-not-adopted grouped variants, ``"no_leading_space"``
    (``"\\d{1,3}"``) and ``"leading_space"`` (``" ?\\d{1,3}"``, which avoids
    wasting a lone-space token before a number), or ``"individual"`` (every
    digit isolated one at a time, via `Digits(individual_digits=True)` -- a
    genuinely different, untested granularity; see `_DIGIT_SPLIT_MODES`'s own
    comment for the real StarCoder2/Llama/Tekken/SmolLM2 precedent). MF-090's
    real comparison found the two grouped variants cost fertility relative to
    no digit-splitting, and none of the three has ever been tested for a
    trained-quality effect (digit-splitting's actual purpose is arithmetic
    capability, not fertility) -- they remain available for a future properly
    isolated test, not for casual use.

    ``pretokenizer`` selects the pre-tokenization regex family: ``"gpt2"``
    (this project's own regex, via `ByteLevel`'s built-in pattern -- Edu's
    real default, `EDU_PRETOKENIZER_MODE`), ``"gpt4"`` (MF-100, the real
    cl100k_base pattern, richer contraction/digit/punctuation handling --
    see `_GPT4_REGEX_PATTERN`'s own comment for the verification trail; a
    real, adopted win for Modern, `MODERN_PRETOKENIZER_MODE`,
    `reports/mf100-stage2-comparison.md`), or ``"o200k"`` (MF-100's wider
    survey, the real GPT-4o/o200k_base pattern -- adds explicit
    case-transition splitting on top of cl100k's own refinements, see
    `_O200K_REGEX_PATTERN`'s comment; won on fertility, lost on real trained
    BPB, not adopted).

    ``"gpt4"``/``"o200k"`` may be combined with a non-``"none"`` ``digit_split``
    (2026-09-12, previously rejected as an untested combination): the digit
    -split stage runs *after* the chosen regex, further splitting whatever
    digit groups that regex already produced (both cap at 1-3 digits
    themselves, so e.g. ``digit_split="individual"`` on top of ``"gpt4"``
    means "gpt4's own richer contraction/whitespace/punctuation splitting,
    but every digit isolated one at a time instead of grouped up to 3" --
    the two axes are independent and compose in the order they run).
    """

    # Every one of the 256 byte values needs a slot, plus the 14 markers, or some
    # inputs would be impossible to encode at all.
    minimum_vocab = len(SPECIAL_TOKENS) + len(ByteLevel.alphabet())
    if vocab_size < minimum_vocab:
        raise ValueError(f"vocab_size must be at least {minimum_vocab} for byte coverage")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if digit_split != "none" and digit_split not in _DIGIT_SPLIT_MODES:
        allowed = sorted(_DIGIT_SPLIT_MODES)
        raise ValueError(f"digit_split must be 'none' or one of {allowed}, got {digit_split!r}")
    if pretokenizer not in ("gpt2", "gpt4", "o200k"):
        raise ValueError(f"pretokenizer must be 'gpt2', 'gpt4', or 'o200k', got {pretokenizer!r}")
    # unk_token=None: with full byte coverage there is no such thing as an unknown
    # character, so an "unknown" token would only ever hide a bug.
    backend = Tokenizer(BPE(unk_token=None))
    if pretokenizer in ("gpt4", "o200k"):
        # The chosen regex runs first; an additional digit-split stage (if
        # requested) further splits whatever digit groups it already produced,
        # then ByteLevel's own GPT-2 regex must stay off, or it would re-split
        # each already-split piece a second time under a third, different rule.
        regex_pattern = _GPT4_REGEX_PATTERN if pretokenizer == "gpt4" else _O200K_REGEX_PATTERN
        stages: list[Any] = [Split(regex_pattern, behavior="isolated")]
        if digit_split == "individual":
            stages.append(Digits(individual_digits=True))
        elif digit_split != "none":
            stages.append(Split(_DIGIT_SPLIT_PATTERNS[digit_split], behavior="isolated"))
        stages.append(ByteLevel(add_prefix_space=False, use_regex=False))
        backend.pre_tokenizer = Sequence(stages)
    else:
        byte_level = ByteLevel(add_prefix_space=False, use_regex=True)
        if digit_split == "none":
            backend.pre_tokenizer = byte_level
        elif digit_split == "individual":
            backend.pre_tokenizer = Sequence([Digits(individual_digits=True), byte_level])
        else:
            backend.pre_tokenizer = Sequence(
                [Split(_DIGIT_SPLIT_PATTERNS[digit_split], behavior="isolated"), byte_level]
            )
    backend.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )
    backend.train_from_iterator(texts, trainer=trainer)
    return MiniFrontierTokenizer(backend, TokenizerContract(vocab_size=vocab_size))
