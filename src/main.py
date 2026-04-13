"""Minimal container entry point for the ML project."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    raw_data_dir = Path("data/raw")
    tracked_files = []

    if raw_data_dir.exists():
        tracked_files = sorted(
            path.name for path in raw_data_dir.iterdir() if path.is_file()
        )

    print("cvd-risk-prediction-mlops")
    print("Container entry point is configured correctly.")
    print("This stub can be replaced with model inference or API startup later.")
    print(f"Raw data directory present: {raw_data_dir.exists()}")
    print(f"Visible files in data/raw: {tracked_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
