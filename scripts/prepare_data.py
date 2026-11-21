"""Convert the raw PaySim CSV to compact parquet and build a small dev sample.

Usage (from the repo root):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --csv "D:/data/PS_2017.csv"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iafd.config import load_config, resolve
from iafd.data import (csv_to_parquet, load_dataset, make_dev_sample,
                       chronological_split, check_split)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(resolve(cfg["paths"]["raw_csv"])))
    args = ap.parse_args()

    csv, pq_path, dev_path = Path(args.csv), resolve(cfg["paths"]["parquet"]), resolve(cfg["paths"]["dev_parquet"])
    if not csv.exists():
        sys.exit(f"CSV not found: {csv}\nDownload PaySim from Kaggle or pass --csv <path>.")

    print(f"Converting {csv} -> {pq_path} (chunked)...")
    n = csv_to_parquet(csv, pq_path)
    print(f"  rows written: {n:,}")

    df = load_dataset(pq_path)
    s = cfg["split"]
    summary = check_split(*chronological_split(df, s["train_end_step"], s["val_end_step"]))
    for name, info in summary.items():
        print(f"  {name:5s} rows={info['rows']:>9,} fraud={info['fraud']:>5,} "
              f"rate={info['fraud_rate']:.4%} steps={info['steps']}")

    dev = make_dev_sample(df, cfg["dev_sample"]["legit_fraction"], cfg["seed"])
    dev.to_parquet(dev_path, index=False)
    print(f"Dev sample: {len(dev):,} rows -> {dev_path}")


if __name__ == "__main__":
    main()
