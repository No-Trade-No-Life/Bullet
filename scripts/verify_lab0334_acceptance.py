#!/usr/bin/env python3
"""Verify one immutable lab-0334 Python-to-Rust production acceptance contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PARITY_PATTERN = re.compile(
    r"parity=pass decisions=(\d+) labels=(\d+) canonical_bytes=(\d+)"
)


class AcceptanceError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != 1:
        raise AcceptanceError("unsupported acceptance contract schema_version")
    return contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tar_member(archive: Path, member_name: str) -> str:
    digest = hashlib.sha256()
    with tarfile.open(archive, "r:gz") as bundle:
        try:
            member = bundle.getmember(member_name)
        except KeyError as error:
            raise AcceptanceError(
                f"{archive} does not contain required member {member_name}"
            ) from error
        source = bundle.extractfile(member)
        if source is None:
            raise AcceptanceError(f"{member_name} is not a regular archive member")
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(name: str, actual: str, expected: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected):
        raise AcceptanceError(f"contract contains invalid SHA-256 for {name}")
    if actual != expected:
        raise AcceptanceError(
            f"{name} SHA-256 mismatch: expected={expected} actual={actual}"
        )


def verify_source_tree(source_tree: Path, expected_commit: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise AcceptanceError(
            f"Bullet source commit mismatch: expected={expected_commit} actual={commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(source_tree), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise AcceptanceError("Bullet source tree has tracked modifications")


def verify_config_data_paths(
    config_path: Path, data_dir: Path, parquet_records: list[dict[str, Any]]
) -> None:
    config = tomllib.loads(config_path.read_text())
    configured = {
        instrument["symbol"]: Path(instrument["parquet"]).expanduser().resolve()
        for instrument in config.get("instruments", [])
    }
    expected = {
        record["symbol"]: (data_dir / record["filename"]).resolve()
        for record in parquet_records
    }
    if configured != expected:
        raise AcceptanceError(
            f"config Parquet mapping mismatch: expected={expected} actual={configured}"
        )


def expected_parity_line(contract: dict[str, Any]) -> str:
    parity = contract["parity"]
    return (
        "parity=pass "
        f"decisions={parity['decisions']} "
        f"labels={parity['labels']} "
        f"canonical_bytes={parity['canonical_bytes']}"
    )


def verify_parity_output(output: str, contract: dict[str, Any]) -> None:
    matches = PARITY_PATTERN.findall(output)
    expected = contract["parity"]
    expected_tuple = (
        str(expected["decisions"]),
        str(expected["labels"]),
        str(expected["canonical_bytes"]),
    )
    if matches != [expected_tuple]:
        raise AcceptanceError(
            f"unexpected parity output: expected={expected_parity_line(contract)!r} "
            f"actual={output.strip()!r}"
        )


def verify_files(arguments: argparse.Namespace, contract: dict[str, Any]) -> None:
    python_reference = contract["python_reference"]
    require_sha256(
        "Python release archive",
        sha256_file(arguments.python_archive),
        python_reference["archive_sha256"],
    )
    require_sha256(
        "Python reference archive member",
        sha256_tar_member(
            arguments.python_archive, python_reference["archive_member"]
        ),
        python_reference["sha256"],
    )
    require_sha256(
        "Python reference",
        sha256_file(arguments.python_reference),
        python_reference["sha256"],
    )

    bullet = contract["bullet"]
    verify_source_tree(arguments.source_tree, bullet["source_commit"])
    require_sha256(
        "Bullet release archive",
        sha256_file(arguments.bullet_release_archive),
        bullet["release_archive_sha256"],
    )
    require_sha256(
        "Bullet release archive member",
        sha256_tar_member(arguments.bullet_release_archive, bullet["release_member"]),
        bullet["binary_sha256"],
    )
    require_sha256(
        "Bullet release artifact",
        sha256_file(arguments.bullet_artifact),
        bullet["binary_sha256"],
    )
    require_sha256(
        "deployed Bullet binary",
        sha256_file(arguments.deployed_binary),
        bullet["binary_sha256"],
    )

    parquet_records = contract["inputs"]["parquet"]
    verify_config_data_paths(arguments.config, arguments.data_dir, parquet_records)
    for record in parquet_records:
        require_sha256(
            f"{record['symbol']} Parquet",
            sha256_file(arguments.data_dir / record["filename"]),
            record["sha256"],
        )

    outputs = contract["reference_outputs"]
    require_sha256(
        "candidate decisions",
        sha256_file(arguments.candidate_decisions),
        outputs["candidate_decisions_sha256"],
    )
    require_sha256(
        "raw candidate labels",
        sha256_file(arguments.raw_candidate_labels),
        outputs["raw_candidate_labels_sha256"],
    )


def run_parity(arguments: argparse.Namespace, contract: dict[str, Any]) -> None:
    completed = subprocess.run(
        [
            str(arguments.deployed_binary),
            "verify-parity",
            str(arguments.config),
            str(arguments.candidate_decisions),
            str(arguments.raw_candidate_labels),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            "Bullet parity process failed: "
            f"exit={completed.returncode} stdout={completed.stdout.strip()!r} "
            f"stderr={completed.stderr.strip()!r}"
        )
    verify_parity_output(completed.stdout, contract)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--python-archive", type=Path, required=True)
    parser.add_argument("--python-reference", type=Path, required=True)
    parser.add_argument("--bullet-release-archive", type=Path, required=True)
    parser.add_argument("--bullet-artifact", type=Path, required=True)
    parser.add_argument("--deployed-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-decisions", type=Path, required=True)
    parser.add_argument("--raw-candidate-labels", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    contract = load_contract(arguments.contract)
    verify_files(arguments, contract)
    run_parity(arguments, contract)
    print(
        "acceptance=pass "
        f"contract={arguments.contract} "
        f"source_commit={contract['bullet']['source_commit']} "
        f"binary_sha256={contract['bullet']['binary_sha256']} "
        f"python_sha256={contract['python_reference']['sha256']}"
    )
    print(expected_parity_line(contract))


if __name__ == "__main__":
    try:
        main()
    except (AcceptanceError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"acceptance=fail {error}") from error
