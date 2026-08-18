"""Frozen byte-level BPE contract and deterministic training/loading helpers."""

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
    """Thin wrapper that validates the immutable MiniFrontier token contract."""

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
    """Train deterministic byte-BPE from an already deterministic text stream."""

    minimum_vocab = len(SPECIAL_TOKENS) + len(ByteLevel.alphabet())
    if vocab_size < minimum_vocab:
        raise ValueError(f"vocab_size must be at least {minimum_vocab} for byte coverage")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
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
