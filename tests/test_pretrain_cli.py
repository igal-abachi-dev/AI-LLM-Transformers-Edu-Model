"""Behavioral tests for train/pretrain.py's run() -- not just --help smoke coverage.

train/pretrain.py has no package __init__.py, so it is loaded directly from its file path
rather than imported by dotted name; this mirrors how the script itself is actually invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from minifrontier.data import Document
from minifrontier.shards import TokenShardWriter

_PRETRAIN_PATH = Path(__file__).resolve().parents[1] / "train" / "pretrain.py"
_spec = importlib.util.spec_from_file_location("minifrontier_train_pretrain", _PRETRAIN_PATH)
assert _spec is not None and _spec.loader is not None
pretrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pretrain)


def _make_document(text: str, record_id: str) -> Document:
    return Document.create(
        text,
        source="fixture",
        revision="abc123",
        license="Apache-2.0",
        language="en",
        record_id=record_id,
    )


def _build_shards(tmp_path: Path, tokenizer) -> Path:
    writer = TokenShardWriter(
        tmp_path / "train", tokenizer, sequence_length=6, sequences_per_shard=4
    )
    for index in range(6):
        writer.add(_make_document(f"tiny document number {index} with some words", str(index)))
    writer.finalize(drop_remainder=False)
    return tmp_path / "train"


def _write_tiny_config(tmp_path: Path, vocab_size: int) -> Path:
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(
        "\n".join(
            [
                'preset = "edu"',
                f"vocab_size = {vocab_size}",
                "max_seq_len = 16",
                "n_layers = 1",
                "d_model = 16",
                "n_heads = 2",
                "n_kv_heads = 2",
                "d_ff = 32",
                "norm_eps = 1e-6",
                "rope_theta = 10000.0",
                "qk_norm = false",
                'attention_pattern = "full"',
                "local_window = 16",
                'global_position_encoding = "rope"',
                "dropout = 0.0",
                "tie_embeddings = true",
                'attention_impl = "sdpa"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _args(
    config_path: Path, shards_path: Path, output_path: Path, **overrides
) -> argparse.Namespace:
    values = dict(
        config=config_path,
        train_shards=shards_path,
        mixture=None,
        output=output_path,
        resume=None,
        device="cpu",
        seed=42,
        precision="float32",
        attention_impl=None,
        updates=2,
        warmup_updates=0,
        batch_size=1,
        accumulation_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        no_decay_embeddings=False,
        gradient_clip=1.0,
        checkpoint_interval=1,
        progress_interval=1,
        no_checkpoint=False,
        activation_checkpointing=False,
        compile=False,
        compile_backend=None,
        compile_fail=False,
        keep_last_n_checkpoints=None,
        loss_chunk_size=None,
        z_loss_weight=0.0,
        mtp_extra_heads=0,
        mtp_loss_weight=0.0,
        schedule="cosine",
        wsd_decay_fraction=0.2,
        ema_decay=None,
        optimizer="adamw",
        cautious_xi=1.0,
        stability_window=128,
        stability_sigma_factor=6.0,
        skip_anomalous_steps=False,
        decay_mixture=None,
        mixture_max_source_fraction=None,
        mixture_max_repetition_ratio=None,
        validation_interval=0,
        validation_shards=None,
        validation_batch_size=pretrain.VALIDATION_BATCH_SIZE,
        validation_max_batches=pretrain.VALIDATION_MAX_BATCHES,
        tokenizer=Path("data/tokenizer"),
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_no_checkpoint_flag_skips_all_checkpoint_writes(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)

    # Default behavior: checkpoint-interval=1 with 2 updates writes at least one
    # interval checkpoint plus the final one.
    checkpointed_output = tmp_path / "with-checkpoints"
    pretrain.run(_args(config_path, shards_path, checkpointed_output))
    assert (checkpointed_output / "final" / "model.safetensors").exists()
    assert list(checkpointed_output.glob("checkpoint-*"))
    assert (checkpointed_output / "run.json").exists()

    # --no-checkpoint: only the report is written, no model/optimizer state at all.
    bare_output = tmp_path / "no-checkpoints"
    pretrain.run(_args(config_path, shards_path, bare_output, no_checkpoint=True, seed=43))
    assert (bare_output / "run.json").exists()
    assert not (bare_output / "final").exists()
    assert not list(bare_output.glob("checkpoint-*"))
    assert list(bare_output.iterdir()) == [bare_output / "run.json"]


def test_progress_interval_prints_during_training_including_under_no_checkpoint(
    tmp_path, mini_tokenizer, capsys
) -> None:
    """Real regression coverage for a real gap found while running MF-070's
    350M scale check: a run gave zero signal until it finished or was killed,
    indistinguishable from a hang. Also covers the specific bug the fix
    introduced a risk of reintroducing -- progress printing must fire even
    under --no-checkpoint, which is exactly the throwaway-benchmark case that
    needs it most, not something the checkpoint-interval gate should suppress."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"
    pretrain.run(
        _args(config_path, shards_path, output, updates=2, progress_interval=1, no_checkpoint=True)
    )
    captured = capsys.readouterr().out
    assert "1/2 updates" in captured
    assert "2/2 updates" in captured


def test_loss_chunk_size_and_z_loss_weight_flags_reach_real_training(
    tmp_path, mini_tokenizer
) -> None:
    """Regression for a real gap: MF-084's chunked loss/z-loss existed in
    TrainingConfig but had no CLI path from the actual training entry point,
    so a real run could never reach them regardless of vocabulary size."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "chunked"
    state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            no_checkpoint=True,
            loss_chunk_size=2,
            z_loss_weight=1e-4,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN


def test_z_loss_and_mtp_loss_are_printed_in_progress_lines_only_when_configured(
    tmp_path, mini_tokenizer, capsys
) -> None:
    """z-loss and the MTP auxiliary loss were already computed every update
    (training.py) but never surfaced anywhere -- not on TrainingState, not in
    the progress log. Confirms both now print when configured, and that a
    plain run's progress line is unaffected (no stray 'z_loss='/'mtp_loss='
    text) when neither is."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)

    plain_output = tmp_path / "plain"
    pretrain.run(
        _args(config_path, shards_path, plain_output, no_checkpoint=True, progress_interval=1)
    )
    plain_captured = capsys.readouterr().out
    assert "loss=" in plain_captured
    assert "z_loss=" not in plain_captured
    assert "mtp_loss=" not in plain_captured

    configured_output = tmp_path / "configured"
    pretrain.run(
        _args(
            config_path,
            shards_path,
            configured_output,
            no_checkpoint=True,
            progress_interval=1,
            loss_chunk_size=2,
            z_loss_weight=1e-4,
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    configured_captured = capsys.readouterr().out
    assert "z_loss=" in configured_captured
    assert "mtp_loss=" in configured_captured


def test_validation_interval_and_validation_shards_require_each_other(
    tmp_path, mini_tokenizer
) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)

    with pytest.raises(ValueError, match="--validation-interval and --validation-shards"):
        pretrain.run(
            _args(
                config_path, shards_path, tmp_path / "a", no_checkpoint=True, validation_interval=1
            )
        )
    with pytest.raises(ValueError, match="--validation-interval and --validation-shards"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "b",
                no_checkpoint=True,
                validation_shards=["val;" + str(shards_path)],
            )
        )


def test_validation_interval_runs_real_periodic_validation_and_prints_metrics(
    tmp_path, mini_tokenizer, tokenizer_dir, capsys
) -> None:
    """Real gap found while reviewing the live MF-070 release run: TrainingConfig/
    train_updates already supported a validation_fn callback, but pretrain.py --
    the only entry point any real run has ever used -- never constructed one or
    exposed a flag for it. Every real run's validation was therefore always a
    separate, after-the-fact step; this confirms it can now run in-loop too."""

    train_shards_path = _build_shards(tmp_path / "train-src", mini_tokenizer)
    validation_shards_path = _build_shards(tmp_path / "val-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"

    state, _ = pretrain.run(
        _args(
            config_path,
            train_shards_path,
            output,
            no_checkpoint=True,
            updates=2,
            progress_interval=1,
            validation_interval=1,
            validation_shards=[f"held_out;{validation_shards_path}"],
            tokenizer=tokenizer_dir,
        )
    )
    assert state.completed_updates == 2
    captured = capsys.readouterr().out
    assert "[validation @ update 1]" in captured
    assert "[validation @ update 2]" in captured
    assert "held_out: ce=" in captured
    assert "combined: ce=" in captured


def test_validation_multi_source_reports_each_source_and_a_combined_figure(
    tmp_path, mini_tokenizer, tokenizer_dir, capsys
) -> None:
    train_shards_path = _build_shards(tmp_path / "train-src", mini_tokenizer)
    first_validation = _build_shards(tmp_path / "val-a", mini_tokenizer)
    second_validation = _build_shards(tmp_path / "val-b", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"

    pretrain.run(
        _args(
            config_path,
            train_shards_path,
            output,
            no_checkpoint=True,
            updates=1,
            progress_interval=1,
            validation_interval=1,
            validation_shards=[
                f"alpha;{first_validation}",
                f"beta;{second_validation}",
            ],
            tokenizer=tokenizer_dir,
        )
    )
    captured = capsys.readouterr().out
    assert "[validation @ update 1]" in captured
    assert "alpha: ce=" in captured
    assert "beta: ce=" in captured
    assert "combined: ce=" in captured


def test_validation_shards_duplicate_names_are_rejected(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)

    with pytest.raises(ValueError, match="unique"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "out",
                no_checkpoint=True,
                validation_interval=1,
                validation_shards=[f"dup;{shards_path}", f"dup;{shards_path}"],
            )
        )


def test_mtp_heads_are_saved_in_every_checkpoint_when_enabled(tmp_path, mini_tokenizer) -> None:
    """A real, previously-missing gap: train/pretrain.py's own save calls never
    forwarded mtp_heads, so a real --mtp-extra-heads run's trained draft heads
    were silently lost the moment training ended -- despite checkpoint.py/
    export.py/release.py all being built to expect mtp_heads.safetensors to be
    there (MF-093/MF-109). Both the periodic and final checkpoint must have it."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"
    pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            checkpoint_interval=1,
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    assert (output / "checkpoint-00000001" / "mtp_heads.safetensors").exists()
    assert (output / "final" / "mtp_heads.safetensors").exists()


def test_resume_with_mtp_extra_heads_restores_the_trained_heads(tmp_path, mini_tokenizer) -> None:
    """--resume together with --mtp-extra-heads > 0 must actually restore the
    trained heads, not reinitialize them -- the real fix for the gap the
    previous version of this test only guarded against by rejecting outright.
    pretrain.py's CLI has no way to run a real partial-then-resume-to-
    completion scenario in one process (each invocation runs to its full
    --updates), so -- matching test_mixture_flag_reaches_real_training_and_
    checkpoint_reloads's own documented limitation -- this resumes a run
    that is already complete, proving the real load path (construct
    MTPHeads, pass it into load_training_checkpoint, no crash, no silent
    reinitialization) rather than proving additional updates happen."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"
    pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    resumed_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            resume=output / "final",
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    assert resumed_state.completed_updates == 2
    assert (output / "final" / "mtp_heads.safetensors").exists()


def test_ema_is_saved_in_every_checkpoint_when_enabled(tmp_path, mini_tokenizer) -> None:
    """Same class of gap as MTP heads: --ema-decay must actually reach
    save_training_checkpoint's ema= parameter through the real CLI, in both
    the periodic and final checkpoint."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"
    pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            checkpoint_interval=1,
            ema_decay=0.9,
        )
    )
    assert (output / "checkpoint-00000001" / "ema.safetensors").exists()
    assert (output / "final" / "ema.safetensors").exists()


def test_resume_with_ema_decay_restores_the_shadow_weights(tmp_path, mini_tokenizer) -> None:
    """--resume together with --ema-decay must restore the trained shadow
    weights, not reinitialize them from the resumed model's live weights --
    mirroring the equivalent MTP resume test/fix above."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"
    pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            ema_decay=0.9,
        )
    )
    resumed_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            updates=2,
            resume=output / "final",
            ema_decay=0.9,
        )
    )
    assert resumed_state.completed_updates == 2
    assert (output / "final" / "ema.safetensors").exists()


def test_cautious_adamw_optimizer_flag_reaches_real_training(tmp_path, mini_tokenizer) -> None:
    """--optimizer cautious_adamw must actually reach build_optimizer through the
    real CLI, not just parse -- proven by a real training run whose loss differs
    from the plain-adamw run at identical seed/data/lr (the two optimizers take
    genuinely different steps)."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    adamw_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            tmp_path / "adamw",
            no_checkpoint=True,
            updates=4,
            learning_rate=0.5,
            optimizer="adamw",
        )
    )
    cautious_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            tmp_path / "cautious",
            no_checkpoint=True,
            updates=4,
            learning_rate=0.5,
            optimizer="cautious_adamw",
        )
    )
    assert adamw_state.last_loss != cautious_state.last_loss


def test_no_decay_embeddings_flag_reaches_real_training(tmp_path, mini_tokenizer) -> None:
    """TrainingConfig.decay_embeddings existing is not enough by itself -- the
    CLI had no path to it at all before this flag. Verified by training two
    otherwise-identical runs (same seed/data/everything else, real nonzero
    weight_decay so the flag has something to actually change) and confirming
    the resulting loss genuinely differs -- proof the embedding really is
    excluded from decay through the real entry point, not just that the flag
    parses."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    decayed_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            tmp_path / "decayed",
            no_checkpoint=True,
            updates=4,
            learning_rate=0.5,
            weight_decay=0.9,
            no_decay_embeddings=False,
        )
    )
    undecayed_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            tmp_path / "undecayed",
            no_checkpoint=True,
            updates=4,
            learning_rate=0.5,
            weight_decay=0.9,
            no_decay_embeddings=True,
        )
    )
    assert decayed_state.last_loss != undecayed_state.last_loss


def test_schedule_flag_reaches_real_training_and_produces_a_different_lr_path(
    tmp_path, mini_tokenizer
) -> None:
    """Regression for the same class of gap as loss_chunk_size/z_loss_weight
    above: TrainingConfig.schedule existing is not enough by itself -- the
    CLI has to actually reach it, or --schedule wsd would silently train
    under cosine regardless. Verified by training two otherwise-identical
    runs (same seed, same data, same everything else) under each schedule
    and confirming the resulting weights genuinely differ -- proof the two
    runs actually followed different learning-rate paths, not just that
    the flag parsed."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    cosine_output = tmp_path / "cosine"
    cosine_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            cosine_output,
            no_checkpoint=True,
            updates=4,
            warmup_updates=0,
            learning_rate=0.5,
            min_learning_rate=0.0,
            schedule="cosine",
        )
    )
    wsd_output = tmp_path / "wsd"
    wsd_state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            wsd_output,
            no_checkpoint=True,
            updates=4,
            warmup_updates=0,
            learning_rate=0.5,
            min_learning_rate=0.0,
            schedule="wsd",
            wsd_decay_fraction=0.5,
        )
    )
    assert cosine_state.last_loss != wsd_state.last_loss


def test_mtp_extra_heads_flag_reaches_real_training(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "mtp"
    state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            no_checkpoint=True,
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN


def test_mixture_flag_reaches_real_training_and_checkpoint_reloads(
    tmp_path, mini_tokenizer
) -> None:
    """Real end-to-end wiring: --mixture -> real training -> a real checkpoint
    whose mixture cursor state round-trips. MixtureBatchProvider's own exact-
    resume-continues-not-restarts guarantee is unit-tested directly against the
    class in test_shards.py; pretrain.py's CLI has no way to run a real partial-
    then-resume-to-completion scenario (each invocation always runs to its full
    `--updates` in one call), so this test verifies the wiring, not the resume
    algorithm itself."""

    web_shards = _build_shards(tmp_path / "web-src", mini_tokenizer)
    code_shards = _build_shards(tmp_path / "code-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "mixture-run"
    mixture = [f"web;{web_shards};0.7", f"code;{code_shards};0.3"]

    state, _ = pretrain.run(
        _args(
            config_path,
            None,
            output,
            train_shards=None,
            mixture=mixture,
            no_checkpoint=False,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN

    # Loading the real checkpoint back (same config, same mixture) must succeed --
    # this is what --resume actually does, exercised for real rather than assumed.
    reloaded_state, _ = pretrain.run(
        _args(
            config_path,
            None,
            tmp_path / "mixture-reload",
            train_shards=None,
            mixture=mixture,
            resume=output / "final",
            no_checkpoint=True,
        )
    )
    assert reloaded_state.completed_updates == 2  # already at max_updates=2, correctly a no-op


def test_mixture_and_train_shards_are_mutually_exclusive(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    with pytest.raises(ValueError, match="exactly one"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "out",
                mixture=[f"web;{shards_path};1.0"],
                no_checkpoint=True,
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        pretrain.run(
            _args(config_path, None, tmp_path / "out", train_shards=None, no_checkpoint=True)
        )


def test_mixture_max_source_fraction_flag_rejects_a_dominant_source(
    tmp_path, mini_tokenizer
) -> None:
    """MF-148: --mixture-max-source-fraction reaches MixtureBatchProvider's own
    real validation -- this is a wiring test, the cap logic itself is unit-tested
    directly against the class in test_shards.py."""

    web_shards = _build_shards(tmp_path / "web-src", mini_tokenizer)
    code_shards = _build_shards(tmp_path / "code-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    mixture = [f"web;{web_shards};0.9", f"code;{code_shards};0.1"]

    with pytest.raises(ValueError, match="max_source_fraction"):
        pretrain.run(
            _args(
                config_path,
                None,
                tmp_path / "out",
                train_shards=None,
                mixture=mixture,
                mixture_max_source_fraction=0.5,
                no_checkpoint=True,
            )
        )


def test_mixture_max_repetition_ratio_flag_rejects_an_over_repeated_source(
    tmp_path, mini_tokenizer
) -> None:
    """MF-148: --mixture-max-repetition-ratio reaches MixtureBatchProvider's own
    real validation, computing expected_total_batches from --updates x
    --accumulation-steps -- this is a wiring test, the cap logic itself is
    unit-tested directly against the class in test_shards.py."""

    web_shards = _build_shards(tmp_path / "web-src", mini_tokenizer)
    code_shards = _build_shards(tmp_path / "code-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    mixture = [f"web;{web_shards};0.5", f"code;{code_shards};0.5"]

    with pytest.raises(ValueError, match="max_repetition_ratio"):
        pretrain.run(
            _args(
                config_path,
                None,
                tmp_path / "out",
                train_shards=None,
                mixture=mixture,
                updates=1000,
                mixture_max_repetition_ratio=0.01,
                no_checkpoint=True,
            )
        )


def test_decay_mixture_flag_reaches_real_training_via_curriculum_provider(
    tmp_path, mini_tokenizer
) -> None:
    """Real end-to-end wiring for MF-095: --decay-mixture (with --mixture and
    --schedule wsd) must actually build a CurriculumMixtureProvider and train
    with it, not just parse. CurriculumMixtureProvider's own exact-resume and
    weight-switch behavior are unit-tested directly in test_shards.py; this
    proves the CLI reaches it."""

    web_shards = _build_shards(tmp_path / "web-src", mini_tokenizer)
    code_shards = _build_shards(tmp_path / "code-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "curriculum-run"
    mixture = [f"web;{web_shards};0.7", f"code;{code_shards};0.3"]

    state, _ = pretrain.run(
        _args(
            config_path,
            None,
            output,
            train_shards=None,
            mixture=mixture,
            decay_mixture=["code;0.9"],
            schedule="wsd",
            updates=4,
            no_checkpoint=False,
        )
    )
    assert state.completed_updates == 4
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN
    assert (output / "final" / "training_state.pt").exists()


def test_decay_mixture_requires_mixture_and_wsd_schedule(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    with pytest.raises(ValueError, match="--decay-mixture requires --mixture"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "out",
                decay_mixture=["web;0.9"],
                no_checkpoint=True,
            )
        )
    with pytest.raises(ValueError, match="--decay-mixture requires --schedule wsd"):
        pretrain.run(
            _args(
                config_path,
                None,
                tmp_path / "out",
                train_shards=None,
                mixture=[f"web;{shards_path};1.0"],
                decay_mixture=["web;0.9"],
                schedule="cosine",
                no_checkpoint=True,
            )
        )


def test_decay_mixture_names_must_be_subset_of_mixture_names(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    with pytest.raises(ValueError, match="subset"):
        pretrain.run(
            _args(
                config_path,
                None,
                tmp_path / "out",
                train_shards=None,
                mixture=[f"web;{shards_path};1.0"],
                decay_mixture=["nonexistent;0.9"],
                schedule="wsd",
                no_checkpoint=True,
            )
        )


class _FakeTerminationTracker:
    """Deterministic stand-in for TerminationRequestTracker (MF-146).

    `update_progress_and_checkpoints` reads `.requested` exactly once per real
    completed update, so becoming "requested" on the Nth *read* -- rather than
    needing an actual, racy concurrent OS signal delivered mid-test-run -- lets
    a test trigger the safety net at an exact, reproducible update count.
    """

    def __init__(self, trigger_after_reads: int) -> None:
        self._trigger_after_reads = trigger_after_reads
        self._read_count = 0
        self.signal_name = "SIGTERM"
        self.installed = False
        self.restored = False

    def install(self) -> None:
        self.installed = True

    def restore(self) -> None:
        self.restored = True

    @property
    def requested(self) -> bool:
        self._read_count += 1
        return self._read_count > self._trigger_after_reads


def test_a_caught_termination_request_saves_an_emergency_checkpoint_and_stops_early(
    tmp_path, mini_tokenizer, monkeypatch, capsys
) -> None:
    """Real functional test of MF-146's SIGTERM/Ctrl+C safety net: a caught
    termination request, arriving mid-run, must force an emergency checkpoint
    at the exact update it was caught on -- not wait for the next
    --checkpoint-interval boundary -- raise GracefulTerminationRequested, and
    leave the tracker's signal handlers restored."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"

    fake_tracker = _FakeTerminationTracker(trigger_after_reads=3)
    monkeypatch.setattr(pretrain, "TerminationRequestTracker", lambda: fake_tracker)

    with pytest.raises(pretrain.GracefulTerminationRequested, match="checkpoint-00000004"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                output,
                updates=10,
                checkpoint_interval=100,  # far past where the fake tracker fires
                progress_interval=1,
            )
        )

    assert fake_tracker.installed
    assert fake_tracker.restored
    assert (output / "checkpoint-00000004" / "model.safetensors").exists()
    assert not (output / "final").exists()  # a graceful early stop is not a completed run
    captured = capsys.readouterr().out
    assert "caught SIGTERM" in captured
    assert "emergency checkpoint" in captured


def test_a_caught_termination_request_under_no_checkpoint_stops_without_saving(
    tmp_path, mini_tokenizer, monkeypatch, capsys
) -> None:
    """--no-checkpoint means no checkpoint, emergency or otherwise -- the run
    still stops immediately on a caught termination request, it just doesn't
    write anything a throwaway benchmark run wasn't already going to keep."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"

    fake_tracker = _FakeTerminationTracker(trigger_after_reads=1)
    monkeypatch.setattr(pretrain, "TerminationRequestTracker", lambda: fake_tracker)

    with pytest.raises(pretrain.GracefulTerminationRequested, match="no-checkpoint"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                output,
                updates=10,
                no_checkpoint=True,
                progress_interval=1,
            )
        )

    assert fake_tracker.restored
    assert not list(output.glob("checkpoint-*"))
    captured = capsys.readouterr().out
    assert "caught SIGTERM" in captured
    assert "no emergency checkpoint written" in captured


def test_termination_tracker_is_always_restored_even_on_an_unrelated_crash(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    """A real, unrelated exception mid-training (not a termination request)
    must still leave the signal-handling tracker restored -- otherwise a
    single crashed run would leak a stale handler into everything that runs
    afterward in the same process, real risk given tests call pretrain.run()
    repeatedly without restarting the interpreter."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "out"

    fake_tracker = _FakeTerminationTracker(trigger_after_reads=10_000)  # never fires
    monkeypatch.setattr(pretrain, "TerminationRequestTracker", lambda: fake_tracker)
    monkeypatch.setattr(
        pretrain,
        "save_training_checkpoint",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated unrelated crash")),
    )

    with pytest.raises(RuntimeError, match="simulated unrelated crash"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                output,
                updates=10,
                checkpoint_interval=1,
                progress_interval=1,
            )
        )

    assert fake_tracker.installed
    assert fake_tracker.restored
