"""Run BERT-Attack in small batches and merge CSV outputs.

Example:
    python attack/run_bertattack_batches.py --total_examples 200 --batch_size 20
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(sys.executable)


def run_batch(args, batch_id, offset, count):
    batch_dir = REPO_ROOT / args.results_dir / "bertattack_batches" / f"batch_{batch_id:03d}_{offset:04d}_{offset + count - 1:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "run.log"

    cmd = [
        str(DEFAULT_PYTHON),
        "attack/baseline_attack.py",
        "--attack",
        "bertattack",
        "--num_examples",
        str(count),
        "--num_examples_offset",
        str(offset),
        "--results_dir",
        str(batch_dir),
        "--example_timeout",
        str(args.example_timeout),
        "--checkpoint_interval",
        str(args.checkpoint_interval),
    ]

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("NLTK_DATA", "/root/nltk_data")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[batch {batch_id}] offset={offset}, count={count}", flush=True)
    print(f"[batch {batch_id}] log={log_path}", flush=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.batch_timeout if args.batch_timeout > 0 else None,
        )
    print(f"[batch {batch_id}] exit_code={proc.returncode}", flush=True)
    return batch_dir / "baseline" / "bertattack_results.csv"


def merge_results(csv_paths, output_csv):
    frames = []
    for csv_path in csv_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.insert(0, "batch_file", str(csv_path))
            frames.append(df)
        else:
            print(f"[WARNING] Missing batch CSV: {csv_path}", flush=True)
    if not frames:
        print("[WARNING] No batch CSV files to merge.", flush=True)
        return
    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(merged)} rows -> {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--example_timeout", type=int, default=300)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument(
        "--batch_timeout",
        type=int,
        default=0,
        help="Optional wall-clock timeout per batch in seconds. 0 disables it.",
    )
    parser.add_argument(
        "--merged_csv",
        default="results/bertattack_batches_merged.csv",
    )
    args = parser.parse_args()

    csv_paths = []
    remaining = args.total_examples
    offset = args.start_offset
    batch_id = 0
    while remaining > 0:
        count = min(args.batch_size, remaining)
        try:
            csv_paths.append(run_batch(args, batch_id, offset, count))
        except subprocess.TimeoutExpired:
            print(f"[ERROR] batch {batch_id} timed out; continuing to next batch.", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] batch {batch_id} failed: {exc}; continuing.", flush=True)
        offset += count
        remaining -= count
        batch_id += 1

    merge_results(csv_paths, REPO_ROOT / args.merged_csv)


if __name__ == "__main__":
    main()
