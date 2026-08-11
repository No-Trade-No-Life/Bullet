#!/usr/bin/env python3
"""Export lab0334's complete pre-arbitration label ledger for Bullet parity."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_lab(path: Path):
    spec = importlib.util.spec_from_file_location("lab0334", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import lab0334 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    arguments = parse_args()
    lab = load_lab(arguments.lab)
    sys.argv = [str(arguments.lab)]
    lab_arguments = lab.parse_args()
    symbols = lab.parse_symbols(lab_arguments.symbols)
    raw = lab.load_data(arguments.data, 0, symbols)
    indicators = {
        symbol: lab.compute_indicators(
            frame,
            lab_arguments.fast_window,
            lab_arguments.slow_window,
            lab_arguments.trend_window,
            lab_arguments.trend_slope_window,
            lab_arguments.min_trend_slope_bps,
            lab_arguments.min_ma_gap_bps,
        )
        for symbol, frame in raw.items()
    }
    base = pd.concat(
        [
            lab.simulate_symbol(indicators[symbol], symbol, lab_arguments)
            for symbol in lab.parse_symbols(lab_arguments.sota_symbols)
        ],
        ignore_index=True,
    )
    base = lab.apply_dynamic_reserve(
        base.sort_values(["symbol", "entry_time", "trade_id"]).reset_index(drop=True),
        lab_arguments,
    )
    candidates = pd.concat(
        [
            base,
            lab.generate_parallel_overlay_candidates(
                indicators,
                lab_arguments,
                len(base) + 1,
            ),
        ],
        ignore_index=True,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(arguments.output, index=False)


if __name__ == "__main__":
    main()
