import pytest
import torch

from minifrontier.chat import (
    ChatMessage,
    encode_chat_prompt,
    fit_messages_to_context,
    generate_assistant,
    load_system_prompt,
    render_chat,
    validate_messages,
)
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.sft import (
    ConversationRecord,
    conversation_hash,
    encode_sft_example,
    pack_sft_examples,
)
from minifrontier.tokenizer import SPECIAL_TOKEN_IDS
from minifrontier.training import ListBatchProvider, TrainingConfig, train_updates


def record(messages: tuple[ChatMessage, ...], record_id: str = "fixture") -> ConversationRecord:
    return ConversationRecord(
        messages=messages,
        source="original-fixture",
        revision="v1",
        license="CC0-1.0",
        record_id=record_id,
        content_hash=conversation_hash(messages),
    )


def test_chat_template_is_deterministic_and_role_order_is_strict() -> None:
    messages = [ChatMessage("system", "Be concise."), ChatMessage("user", "Hi")]
    assert render_chat(messages, add_generation_prompt=True) == (
        "<|bos|><|system|>\nBe concise.<|eos|>\n<|user|>\nHi<|eos|>\n<|assistant|>\n"
    )
    with pytest.raises(ValueError, match="expected assistant"):
        validate_messages([ChatMessage("user", "one"), ChatMessage("user", "two")])
    with pytest.raises(ValueError, match="end with a user"):
        validate_messages(
            [ChatMessage("user", "one"), ChatMessage("assistant", "two")],
            generation_prompt=True,
        )


def test_default_system_prompt_is_compact_and_capability_honest() -> None:
    prompt = load_system_prompt()
    assert "general-purpose and coding assistant" in prompt
    assert "Never claim to have" in prompt
    assert "tools" in prompt
    assert len(prompt.split()) < 180


def test_assistant_only_mask_excludes_system_user_and_role_tokens(mini_tokenizer) -> None:
    messages = (
        ChatMessage("system", "rules"),
        ChatMessage("user", "question"),
        ChatMessage("assistant", "answer"),
    )
    example = encode_sft_example(record(messages), mini_tokenizer, max_length=128)
    assistant_role = example.token_ids.index(SPECIAL_TOKEN_IDS["<|assistant|>"])
    user_role = example.token_ids.index(SPECIAL_TOKEN_IDS["<|user|>"])
    assert not any(example.loss_mask[: assistant_role + 1])
    assert example.loss_mask[user_role] is False
    assert any(example.loss_mask[assistant_role + 1 :])
    assert example.loss_mask[example.token_ids.index(mini_tokenizer.eos_id, assistant_role)]


def test_jinja_chat_runtime_and_sft_serialization_share_one_token_contract(
    mini_tokenizer,
) -> None:
    messages = (
        ChatMessage("system", "Be accurate."),
        ChatMessage("user", "First question"),
        ChatMessage("assistant", "First answer"),
        ChatMessage("user", "Second question"),
        ChatMessage("assistant", "Second answer"),
    )
    rendered_training = mini_tokenizer.encode(render_chat(messages, add_generation_prompt=False))
    runtime_training = encode_chat_prompt(
        mini_tokenizer,
        messages,
        add_generation_prompt=False,
    )
    example = encode_sft_example(record(messages), mini_tokenizer, max_length=128)
    assert rendered_training == runtime_training == list(example.token_ids)

    prompt_messages = messages[:-1]
    assert mini_tokenizer.encode(
        render_chat(prompt_messages, add_generation_prompt=True)
    ) == encode_chat_prompt(
        mini_tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )


def test_sft_truncation_drops_only_complete_old_pairs(mini_tokenizer) -> None:
    messages = (
        ChatMessage("system", "rules"),
        ChatMessage("user", "old question"),
        ChatMessage("assistant", "old answer"),
        ChatMessage("user", "new question"),
        ChatMessage("assistant", "new answer"),
    )
    shortened = record((messages[0], messages[3], messages[4]), "short")
    short_example = encode_sft_example(shortened, mini_tokenizer, max_length=128)
    truncated = encode_sft_example(
        record(messages),
        mini_tokenizer,
        max_length=len(short_example.token_ids),
    )
    assert truncated.token_ids == short_example.token_ids
    assert truncated.loss_mask == short_example.loss_mask


def test_sft_packing_preserves_masks_and_padding(mini_tokenizer) -> None:
    messages = (ChatMessage("user", "q"), ChatMessage("assistant", "a"))
    example = encode_sft_example(record(messages), mini_tokenizer, max_length=64)
    sequence_length = len(example.token_ids) + 5
    packed = list(
        pack_sft_examples(
            [example],
            sequence_length=sequence_length,
            pad_id=mini_tokenizer.pad_id,
        )
    )
    assert len(packed) == 1
    assert packed[0].tokens.shape == packed[0].loss_mask.shape == (1, sequence_length)
    assert not packed[0].loss_mask[0, -5:].any()
    assert packed[0].tokens[0, -5:].eq(mini_tokenizer.pad_id).all()


def test_chat_context_drops_oldest_complete_pair(mini_tokenizer) -> None:
    messages = [
        ChatMessage("system", "rules"),
        ChatMessage("user", "old"),
        ChatMessage("assistant", "answer"),
        ChatMessage("user", "latest"),
    ]
    expected = [messages[0], messages[3]]
    budget = len(encode_chat_prompt(mini_tokenizer, expected, add_generation_prompt=True))
    kept, token_ids = fit_messages_to_context(
        mini_tokenizer,
        messages,
        max_prompt_tokens=budget,
    )
    assert kept == expected
    assert len(token_ids) == budget


def test_template_aware_chat_generation_cpu_smoke(mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(
        vocab_size=max(512, mini_tokenizer.vocab_size),
        max_seq_len=64,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    model = MiniFrontier(config).eval()
    result = generate_assistant(
        model,
        mini_tokenizer,
        [ChatMessage("user", "Hi")],
        max_new_tokens=2,
        seed=7,
    )
    assert isinstance(result, str)


@pytest.mark.slow
def test_tiny_assistant_only_sft_overfits(mini_tokenizer) -> None:
    torch.manual_seed(58)
    config = ModelConfig.tiny_edu(
        vocab_size=max(512, mini_tokenizer.vocab_size),
        max_seq_len=32,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    model = MiniFrontier(config)
    messages = (ChatMessage("user", "2+2?"), ChatMessage("assistant", "4"))
    example = encode_sft_example(record(messages), mini_tokenizer, max_length=32)
    batch = next(
        pack_sft_examples(
            [example],
            sequence_length=32,
            pad_id=mini_tokenizer.pad_id,
        )
    )
    training = TrainingConfig(
        max_updates=120,
        learning_rate=2e-2,
        min_learning_rate=2e-3,
        warmup_updates=5,
        weight_decay=0.0,
        gradient_clip=10.0,
    )
    _, _, state, _ = train_updates(model, ListBatchProvider([batch]), training)
    assert state.last_loss is not None and state.last_loss < 0.02
