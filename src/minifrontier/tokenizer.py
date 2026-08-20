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
tokenizer trained once can be compared across experiments. The first eleven IDs
are reserved for markers that never appear in ordinary text::

    <|pad|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>
    <|fim_prefix|> <|fim_suffix|> <|fim_middle|> <|tool_call|> <|tool_result|>

Those markers are the entire mechanism behind chat roles. There is no "assistant
mode" inside the model -- just a token that means the assistant's turn starts
here. It is also why prompt injection is possible at all: text that smuggles in
convincing markers can be read as structure rather than content.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Frozen for the whole project. Commercial models use 100k-200k; the idea is the
# same, and a smaller vocabulary keeps the tied embedding table affordable here.
VOCAB_SIZE: Final = 16_384
TOKENIZER_VERSION: Final = 1
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
)
# IDs 0..10, assigned by position: <|pad|> is 0, <|bos|> is 1, and so on. These
# are part of the frozen contract, because a checkpoint trained with <|eos|> as 2
# produces nonsense if reloaded against a tokenizer that numbered them otherwise.
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

    def _validate_special_tokens(self) -> None:
        """Refuse a tokenizer whose marker IDs drifted -- a silent, ruinous mismatch."""

        for token, expected_id in SPECIAL_TOKEN_IDS.items():
            actual_id = self.backend.token_to_id(token)
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


def train_byte_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
) -> MiniFrontierTokenizer:
    """Train deterministic byte-BPE from an already deterministic text stream.

    "Training" a tokenizer is nothing like training the model -- there is no
    gradient here. It just counts which adjacent pairs of symbols occur most often
    in the corpus and merges them, over and over, until the vocabulary is full.

    The order of ``texts`` affects the merge counts, so the caller is responsible
    for handing over a stream that is already deterministic; otherwise two "same"
    tokenizers would disagree about token IDs.
    """

    # Every one of the 256 byte values needs a slot, plus the 11 markers, or some
    # inputs would be impossible to encode at all.
    minimum_vocab = len(SPECIAL_TOKENS) + len(ByteLevel.alphabet())
    if vocab_size < minimum_vocab:
        raise ValueError(f"vocab_size must be at least {minimum_vocab} for byte coverage")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    # unk_token=None: with full byte coverage there is no such thing as an unknown
    # character, so an "unknown" token would only ever hide a bug.
    backend = Tokenizer(BPE(unk_token=None))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
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
