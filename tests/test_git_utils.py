import subprocess
import time
from pathlib import Path

from minifrontier.git_utils import (
    GITHUB_CACHE_STALENESS_MARKER,
    clone_via_cached_mirror,
    ensure_repo_mirror,
    rmtree_readonly_safe,
)


def _init_local_git_repo(path: Path, *, files: dict[str, str]) -> str:
    """A real, local, fully offline git repo -- used as a fake "origin" so
    the real MF-134 mirror-cache functions can be tested against real git
    subprocess behavior without any network access. Returns the real commit SHA.
    """
    subprocess.run(["git", "init", "--quiet", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True
    )
    for relative_path, content in files.items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "--quiet", "-m", "commit"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_local_git_repo(path: Path, *, files: dict[str, str]) -> str:
    for relative_path, content in files.items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "--quiet", "-m", "second commit"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_ensure_repo_mirror_bootstraps_a_real_fresh_mirror(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    first_sha = _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    mirror_path = ensure_repo_mirror(
        "x/a",
        cache_root,
        max_staleness_seconds=3600,
        force_refresh=False,
        clone_url=str(origin),
    )
    assert mirror_path.exists()
    assert (mirror_path / GITHUB_CACHE_STALENESS_MARKER).exists()
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == first_sha


def test_ensure_repo_mirror_reuses_within_staleness_window_without_fetching(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    first_sha = _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    _commit_local_git_repo(origin, files={"b.py": "pass\n"})
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == first_sha


def test_ensure_repo_mirror_fetches_real_new_commits_once_stale(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    (mirror_path / GITHUB_CACHE_STALENESS_MARKER).write_text(
        str(time.time() - 1_000_000), encoding="utf-8"
    )
    second_sha = _commit_local_git_repo(origin, files={"b.py": "pass\n"})
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == second_sha


def test_ensure_repo_mirror_preserves_the_existing_cache_when_a_refresh_fails(tmp_path) -> None:
    """A real, deliberate design decision (2026-09-15, user-confirmed): a
    repo that disappears upstream (renamed/deleted/gone private) is never
    removed from `configs/code-repo-allowlist.txt` automatically, and a
    failed refresh must never delete the existing mirror -- the last real,
    successfully-fetched content stays available and keeps being used
    (`_extract_repo_documents`'s own outer try/except is what makes the
    *current run* skip it gracefully; this test covers the other real half
    of the guarantee, that the cache itself survives untouched)."""

    origin = tmp_path / "origin"
    origin.mkdir()
    real_sha = _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    (mirror_path / GITHUB_CACHE_STALENESS_MARKER).write_text(
        str(time.time() - 1_000_000), encoding="utf-8"
    )

    # The repo is now gone -- the exact real scenario this decision covers.
    rmtree_readonly_safe(origin)

    try:
        ensure_repo_mirror(
            "x/a",
            cache_root,
            max_staleness_seconds=3600,
            force_refresh=False,
            clone_url=str(origin),
        )
        raised = False
    except subprocess.CalledProcessError:
        raised = True
    assert raised, "a fetch against a now-gone repo must fail loudly, not silently succeed"

    # The real point: the existing mirror is untouched, still the last real,
    # successfully-fetched content -- not deleted just because a refresh failed.
    assert mirror_path.exists()
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == real_sha


def test_ensure_repo_mirror_self_heals_a_leftover_staging_dir_from_an_interrupted_clone(
    tmp_path,
) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    real_sha = _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    # Simulate a hard kill mid-clone: a `<repo>.git.tmp` staging directory
    # exists (with bogus content), but the real `<repo>.git` mirror was
    # never published (matches what the real staging-then-rename code
    # leaves behind if interrupted before the rename).
    staging_path = cache_root / "x__a.git.tmp"
    staging_path.mkdir()
    (staging_path / "bogus").write_text("not a real git repo", encoding="utf-8")
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    assert not staging_path.exists()
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == real_sha


def test_ensure_repo_mirror_force_refresh_reclones_regardless_of_staleness(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_local_git_repo(origin, files={"a.py": "pass\n"})
    cache_root = tmp_path / "cache"
    ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    second_sha = _commit_local_git_repo(origin, files={"b.py": "pass\n"})
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=True, clone_url=str(origin)
    )
    result = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == second_sha


def test_clone_via_cached_mirror_yields_real_files_and_matching_commit_sha(tmp_path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    real_sha = _init_local_git_repo(origin, files={"a.py": "print('hi')\n"})
    destination = tmp_path / "checkout"
    result_sha = clone_via_cached_mirror(
        "x/a",
        destination,
        cache_dir=tmp_path / "cache",
        force_refresh=False,
        max_staleness_seconds=3600,
        clone_url=str(origin),
    )
    assert result_sha == real_sha
    assert (destination / "a.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_ensure_repo_mirror_follows_a_real_default_branch_rename_on_refresh(tmp_path) -> None:
    """MF-134 follow-up (2026-09-15): `--mirror`'s own fetch refspec
    (`+refs/*:refs/*`) does not cover the top-level `HEAD` file, so a plain
    `git fetch --prune` alone never re-points a stale mirror at a renamed
    remote default branch -- real, verified against git's own docs. The fix
    (`git ls-remote --symref` + `git symbolic-ref HEAD`, `_sync_mirror_head`)
    is exercised here for real: the origin's default branch is renamed after
    the mirror is first bootstrapped, and a refresh must follow it.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_local_git_repo(origin, files={"a.py": "pass\n"})
    original_branch = subprocess.run(
        ["git", "-C", str(origin), "symbolic-ref", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cache_root = tmp_path / "cache"
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    initial_head = subprocess.run(
        ["git", "-C", str(mirror_path), "symbolic-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert initial_head == f"refs/heads/{original_branch}"

    # Real rename on the origin, matching what a repo maintainer renaming
    # their default branch (e.g. master -> main) actually does.
    new_sha = _commit_local_git_repo(origin, files={"b.py": "pass\n"})
    subprocess.run(
        ["git", "-C", str(origin), "branch", "-m", original_branch, "renamed-default"],
        check=True,
        capture_output=True,
    )

    (mirror_path / GITHUB_CACHE_STALENESS_MARKER).write_text(
        str(time.time() - 1_000_000), encoding="utf-8"
    )
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )

    updated_head = subprocess.run(
        ["git", "-C", str(mirror_path), "symbolic-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert updated_head == "refs/heads/renamed-default"
    resolved_sha = subprocess.run(
        ["git", "-C", str(mirror_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved_sha == new_sha


def test_ensure_repo_mirror_prunes_a_real_deleted_tag_on_refresh(tmp_path) -> None:
    """--prune-tags (2026-09-15 addition): plain --prune alone does not
    remove deleted tags by default -- verified against real git-mirroring
    guidance, tested here directly rather than trusted from a doc summary.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_local_git_repo(origin, files={"a.py": "pass\n"})
    subprocess.run(["git", "-C", str(origin), "tag", "v1.0"], check=True, capture_output=True)
    cache_root = tmp_path / "cache"
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    tags_before = subprocess.run(
        ["git", "-C", str(mirror_path), "tag"], check=True, capture_output=True, text=True
    ).stdout.split()
    assert "v1.0" in tags_before

    subprocess.run(["git", "-C", str(origin), "tag", "-d", "v1.0"], check=True, capture_output=True)
    (mirror_path / GITHUB_CACHE_STALENESS_MARKER).write_text(
        str(time.time() - 1_000_000), encoding="utf-8"
    )
    mirror_path = ensure_repo_mirror(
        "x/a", cache_root, max_staleness_seconds=3600, force_refresh=False, clone_url=str(origin)
    )
    tags_after = subprocess.run(
        ["git", "-C", str(mirror_path), "tag"], check=True, capture_output=True, text=True
    ).stdout.split()
    assert "v1.0" not in tags_after


def test_clone_via_cached_mirror_falls_back_to_ephemeral_clone_when_cache_dir_is_none(
    monkeypatch,
) -> None:
    calls = []

    def fake_ephemeral(repo_name, destination):
        calls.append((repo_name, destination))
        return "fake-sha"

    monkeypatch.setattr("minifrontier.git_utils.clone_repo_ephemeral", fake_ephemeral)
    result = clone_via_cached_mirror(
        "x/a",
        Path("unused"),
        cache_dir=None,
        force_refresh=False,
        max_staleness_seconds=3600,
    )
    assert result == "fake-sha"
    assert calls == [("x/a", Path("unused"))]
