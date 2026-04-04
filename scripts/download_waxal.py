"""Download Google WAXAL Kikuyu speech dataset."""
import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download WAXAL Kikuyu subset")
    parser.add_argument("--output", default="data/waxal_kikuyu", help="Output directory")
    parser.add_argument("--split", default="train", help="Dataset split")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading WAXAL Kikuyu dataset from HuggingFace...")
    # NOTE: Update the dataset ID once the official WAXAL HuggingFace repo is confirmed
    ds = load_dataset("google/waxal", "kikuyu", split=args.split, cache_dir=str(out / ".cache"))
    ds.save_to_disk(str(out / args.split))
    print(f"Saved {len(ds)} samples to {out / args.split}")


if __name__ == "__main__":
    main()
