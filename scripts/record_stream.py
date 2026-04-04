"""Record Kameme FM live stream audio."""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Record Kameme FM stream")
    parser.add_argument("--duration", type=int, default=3600, help="Recording duration in seconds")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir or cfg["data"]["kameme_raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"kameme_{timestamp}.wav"

    url = cfg["stream"]["url"]
    sr = cfg["audio"]["target_sample_rate"]

    print(f"Recording {url} for {args.duration}s → {out_file}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", url,
        "-t", str(args.duration),
        "-ar", str(sr),
        "-ac", "1",
        str(out_file),
    ], check=True)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
