# lab-0334 production acceptance

## Result

The accepted production snapshot is an exact candidate-ledger and label-ledger match between the E-Works Python strategy truth and the deployed Bullet binary:

```text
parity=pass decisions=5395 labels=5395 canonical_bytes=5839749
```

This result applies only to the immutable files and versions in [`contracts/lab0334-production-2026-08-24.json`](../contracts/lab0334-production-2026-08-24.json). Similar returns, positions, trade counts, or final equity are not parity evidence.

## Acceptance boundary

The contract pins all artifacts that participated in the passing comparison:

- E-Works release archive and `run_optimized.py` content hashes;
- the observed Python interpreter and imported package versions;
- all four Parquet content hashes, row counts, and the `Asia/Shanghai` timestamp schema;
- Bullet source commit, release archive, extracted artifact, and deployed binary hashes;
- Python candidate-decision and raw-label CSV hashes;
- expected decision count, label count, and canonical byte count.

The production Python reference is the release member with SHA-256 `bb1451b6…c6f5`. The older E-Works Git-history file at commit `91998c6…` has SHA-256 `47c495f9…28b3` and cannot consume the current timezone-aware Parquet without failing. It is not the production truth for this acceptance.

The environment record pins the interpreter and packages observed during the passing Python run: Python 3.12.7, pandas 2.2.2, NumPy 1.26.4, XGBoost 3.2.0, and PyArrow 16.1.0. The original release did not include a complete transitive `pip freeze`; a future Python reference release must include that file inside its immutable archive. The current candidate and label CSV hashes remain the output truth even if that historical environment can no longer be reconstructed.

## Re-run the gate

Run the verifier where the exact source worktree, release archives, extracted artifact, deployed binary, production config, frozen Parquet, and Python outputs are available:

```bash
python3 scripts/verify_lab0334_acceptance.py \
  --contract contracts/lab0334-production-2026-08-24.json \
  --source-tree /tmp/Bullet-4051218 \
  --python-archive /tmp/eworks-lab0334-final-full-parity.tar.gz \
  --python-reference /tmp/eworks/labs/lab-0334/run_optimized.py \
  --bullet-release-archive /tmp/bullet-live-x86_64-unknown-linux-musl.tgz \
  --bullet-artifact /tmp/bullet-live-x86_64-unknown-linux-musl \
  --deployed-binary /opt/bullet/bullet-live \
  --config /etc/bullet/lab0334-live.toml \
  --data-dir /var/lib/bullet \
  --candidate-decisions /tmp/candidate_decisions.csv \
  --raw-candidate-labels /tmp/raw_candidate_labels.csv
```

The command fails before parity if any source commit, archive member, file hash, config-to-Parquet mapping, or reference output differs. It then runs the exact deployed binary's `verify-parity` command and requires the declared counts and canonical byte count. Success prints both `acceptance=pass` and the exact `parity=pass` line.

The verifier reads public configuration only. It does not read token files, mutate Parquet, modify account state, or contact the live service.

## CI boundary

`pr-check` runs the verifier's standard-library tests. The tests prove that the gate rejects a wrong Python reference, Parquet file, Bullet source commit, Bullet artifact, config mapping, and parity summary.

GitHub Actions cannot execute the complete production acceptance because this repository has no AWS credentials or private production artifacts. Full acceptance is therefore an explicit release/deployment gate, not a synthetic CI substitute. A new Bullet source commit or binary is not production-equivalent until a new immutable contract and complete `acceptance=pass` record exist.

## Preserved production evidence

The read-only production audit on August 24, 2026 recorded:

- SSM parity command: `a380c9b3-8059-407e-ab78-e6efb86a59e9`;
- deployed binary SHA-256: `712abfdf30cba791ca85a7c9e19d034f7f9ab2f1a3e93b9bc8596126eddb310a`;
- Bullet source commit: `40512180824e22ec49109abf9ed0bf56eb83f187`;
- Python reference SHA-256: `bb1451b69fe6ddb0c6e50510dda6f8d13612dc8b336428c2f208de16e994c6f5`;
- report SHA-256: `bb7b9533b647f8db580737aae3d4ba3e58d4e7a69f611152cbc30f766bb3e4ef`.

The temporary parity files were removed after verification. `bullet-live.service` and `caddy.service` remained active.

## Controlled performance evidence

SSM command `7fd0dc58-51d0-45e3-929e-4df67062c974` ran bounded, read-only probes against the accepted deployed binary on the production `t3a.small` instance: two AMD EPYC 7571 vCPUs, 1.9 GiB RAM, Amazon Linux 2023, with no swap. The live service remained active throughout.

Five 20,000-event in-process tick-to-target samples produced:

| Sample | p50 | p99 | max | Peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 6.550 µs | 34.450 µs | 428.193 µs | 3,468 KiB |
| 2 | 6.500 µs | 34.081 µs | 748.936 µs | 3,464 KiB |
| 3 | 6.510 µs | 39.241 µs | 509.074 µs | 3,468 KiB |
| 4 | 6.530 µs | 36.411 µs | 541.265 µs | 3,468 KiB |
| 5 | 6.540 µs | 38.330 µs | 1,459.533 µs | 3,464 KiB |

The worst observed p99 was 39.241 µs, below the declared 100 ms gate. This benchmark is in-process latency; it does not include CTPD network transit, scheduling before process receipt, or downstream API polling.

Three full four-model history reconstructions with `history_seed_bars=1000000` produced:

| Sample | Elapsed | Peak RSS |
|---:|---:|---:|
| 1 | 11.792 s | 192,784 KiB |
| 2 | 12.051 s | 192,780 KiB |
| 3 | 12.043 s | 192,780 KiB |

The running service's observed high-water RSS was 192,444 KiB and its post-probe RSS was 10,716 KiB. These measurements establish a controlled baseline; they do not change strategy semantics and are not profitability evidence.
