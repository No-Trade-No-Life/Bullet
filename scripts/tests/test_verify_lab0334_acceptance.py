from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "verify_lab0334_acceptance.py"
SPEC = importlib.util.spec_from_file_location("acceptance", SCRIPT)
assert SPEC and SPEC.loader
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AcceptanceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.email", "test@example.test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.name", "Acceptance Test"],
            check=True,
        )
        (self.source / "tracked").write_text("source\n")
        subprocess.run(["git", "-C", str(self.source), "add", "tracked"], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "commit", "-qm", "fixture"], check=True
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.python_reference = root / "run_optimized.py"
        self.python_reference.write_bytes(b"python truth\n")
        self.python_archive = root / "python.tar.gz"
        self._archive(
            self.python_archive,
            self.python_reference,
            "labs/lab-0334/run_optimized.py",
        )

        self.bullet_artifact = root / "bullet-live"
        self.bullet_artifact.write_bytes(b"rust artifact\n")
        self.deployed_binary = root / "deployed-bullet-live"
        self.deployed_binary.write_bytes(self.bullet_artifact.read_bytes())
        self.bullet_archive = root / "bullet.tgz"
        self._archive(
            self.bullet_archive,
            self.bullet_artifact,
            "bullet-live-x86_64-unknown-linux-musl",
        )

        self.data_dir = root / "data"
        self.data_dir.mkdir()
        self.parquet = self.data_dir / "IF8888.parquet"
        self.parquet.write_bytes(b"parquet fixture\n")
        self.config = root / "config.toml"
        self.config.write_text(
            "[[instruments]]\n"
            'symbol = "IF8888"\n'
            f'parquet = "{self.parquet}"\n'
        )
        self.decisions = root / "candidate_decisions.csv"
        self.decisions.write_bytes(b"decisions\n")
        self.labels = root / "raw_candidate_labels.csv"
        self.labels.write_bytes(b"labels\n")

        self.contract = {
            "schema_version": 1,
            "python_reference": {
                "archive_sha256": acceptance.sha256_file(self.python_archive),
                "archive_member": "labs/lab-0334/run_optimized.py",
                "sha256": digest(self.python_reference.read_bytes()),
            },
            "inputs": {
                "parquet": [
                    {
                        "symbol": "IF8888",
                        "filename": self.parquet.name,
                        "sha256": digest(self.parquet.read_bytes()),
                    }
                ]
            },
            "bullet": {
                "source_commit": self.commit,
                "release_archive_sha256": acceptance.sha256_file(
                    self.bullet_archive
                ),
                "release_member": "bullet-live-x86_64-unknown-linux-musl",
                "binary_sha256": digest(self.bullet_artifact.read_bytes()),
            },
            "reference_outputs": {
                "candidate_decisions_sha256": digest(self.decisions.read_bytes()),
                "raw_candidate_labels_sha256": digest(self.labels.read_bytes()),
            },
            "parity": {"decisions": 2, "labels": 2, "canonical_bytes": 42},
        }
        self.arguments = argparse.Namespace(
            source_tree=self.source,
            python_archive=self.python_archive,
            python_reference=self.python_reference,
            bullet_release_archive=self.bullet_archive,
            bullet_artifact=self.bullet_artifact,
            deployed_binary=self.deployed_binary,
            config=self.config,
            data_dir=self.data_dir,
            candidate_decisions=self.decisions,
            raw_candidate_labels=self.labels,
        )

    @staticmethod
    def _archive(archive: Path, source: Path, name: str) -> None:
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source, arcname=name)


class VerifyAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = AcceptanceFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_complete_matching_contract(self) -> None:
        acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_python_reference(self) -> None:
        self.fixture.python_reference.write_bytes(b"wrong python\n")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "Python reference"):
            acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_parquet(self) -> None:
        self.fixture.parquet.write_bytes(b"wrong parquet\n")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "IF8888 Parquet"):
            acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_bullet_source(self) -> None:
        self.fixture.contract["bullet"]["source_commit"] = "0" * 40
        with self.assertRaisesRegex(acceptance.AcceptanceError, "source commit"):
            acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_bullet_artifact(self) -> None:
        self.fixture.bullet_artifact.write_bytes(b"wrong artifact\n")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "release artifact"):
            acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_config_mapping(self) -> None:
        self.fixture.config.write_text(
            "[[instruments]]\n"
            'symbol = "IF8888"\n'
            f'parquet = "{self.fixture.root / "other.parquet"}"\n'
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "mapping mismatch"):
            acceptance.verify_files(self.fixture.arguments, self.fixture.contract)

    def test_rejects_wrong_parity_summary(self) -> None:
        with self.assertRaisesRegex(acceptance.AcceptanceError, "unexpected parity"):
            acceptance.verify_parity_output(
                "parity=pass decisions=2 labels=2 canonical_bytes=41\n",
                self.fixture.contract,
            )

    def test_accepts_exact_parity_summary(self) -> None:
        acceptance.verify_parity_output(
            "parity=pass decisions=2 labels=2 canonical_bytes=42\n",
            self.fixture.contract,
        )


if __name__ == "__main__":
    unittest.main()
