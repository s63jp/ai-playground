"""Protected M6-B comparison baselines (baselines/ at the repository root).

These are read-only reference brains (B3, B4). The tests below enforce that:
  * every file's bytes match the SHA-256 recorded in baselines/manifest.json;
  * each checkpoint's internal content_sha256 still verifies;
  * the folder holds exactly the manifest's files (nothing missing or added);
  * the manifest uses repository-relative paths;
  * the checkpoints are legacy contract_version 3 (43 obs, 14 actions);
  * loading a baseline never writes to it;
  * no code under playground/ or scripts/ points a write at baselines/.

The hashing tests need only the standard library. The loader test needs torch,
which CI installs. No MuJoCo or XPU is used, so this runs on Ubuntu CI.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "baselines"
MANIFEST = BASE / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _entries():
    for brain, b in _manifest()["brains"].items():
        for f in b["files"]:
            yield brain, b, f


def test_gitattributes_keeps_baseline_bytes_exact():
    lines = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "baselines/** -text" in lines


def test_manifest_paths_are_repository_relative():
    for _, b, f in _entries():
        for rel in (f["path"], b["final_checkpoint"]):
            assert not Path(rel).is_absolute(), rel
            assert ":" not in rel and not rel.startswith(("/", "\\")), rel


def test_every_file_matches_manifest_sha256_and_size():
    for brain, _, f in _entries():
        p = BASE / f["path"]
        assert p.is_file(), f"missing: {f['path']}"
        assert p.stat().st_size == f["size_bytes"], f"size changed: {f['path']}"
        assert _sha256(p) == f["sha256"], f"bytes changed: {f['path']}"


def test_folder_holds_exactly_the_manifest_files():
    expected = {f["path"] for _, _, f in _entries()} | {"manifest.json", "README.md"}
    on_disk = {p.relative_to(BASE).as_posix() for p in BASE.rglob("*") if p.is_file()}
    assert on_disk == expected, (
        f"extra: {sorted(on_disk - expected)}; missing: {sorted(expected - on_disk)}")


def test_checkpoint_internal_hashes_and_legacy_contract():
    for brain, b in _manifest()["brains"].items():
        body = json.loads((BASE / b["final_checkpoint"]).read_text(encoding="utf-8"))
        inner = {k: v for k, v in body.items() if k != "content_sha256"}
        recomputed = hashlib.sha256(json.dumps(inner, sort_keys=True).encode()).hexdigest()
        assert recomputed == body["content_sha256"] == b["content_sha256"], brain
        c = body["meta"]["contract"]
        assert c["contract_version"] == 3 == b["contract_version"], brain
        assert (c["obs_dim"], c["action_dim"]) == (43, 14), brain


def test_loading_baselines_never_writes_them():
    pytest.importorskip("torch")
    from playground.rl.checkpoint import load_checkpoint

    before = {p: _sha256(p) for p in BASE.rglob("*") if p.is_file()}
    for brain, b in _manifest()["brains"].items():
        path = BASE / b["final_checkpoint"]
        c = json.loads(path.read_text(encoding="utf-8"))["meta"]["contract"]
        ck = load_checkpoint(path, obs_fingerprint=c["observation"],
                             action_fingerprint=c["action"])
        assert ck["content_sha256"] == b["content_sha256"], brain
    after = {p: _sha256(p) for p in BASE.rglob("*") if p.is_file()}
    assert after == before, "loading a baseline modified the protected folder"


def test_no_code_points_a_write_at_baselines():
    # Forbid literal references to the protected folder in playground/ and scripts/:
    #   REPO/"baselines", "baselines/" or "baselines\\" path literals, experiments/baselines.
    pattern = re.compile(
        r"""(REPO|ROOT)\s*/\s*["']baselines["']|["']baselines[/\\]|experiments[/\\]baselines""")
    offenders = []
    for root in ("playground", "scripts"):
        for py in (REPO / root).rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{py.relative_to(REPO).as_posix()}:{n}")
    assert not offenders, f"code references the protected baselines folder: {offenders}"
