"""
Module: Dataset Explorer

Purpose:
Explore the CICIDS2017 dataset.

Author: Mohammed Uzair Shaikh
Project: Privacy-Preserving Federated Network Intrusion Detection System
"""

from pathlib import Path


def get_dataset_path() -> Path:
    """Return the CICIDS2017 dataset folder path relative to this file."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "CICIDS2017"


def list_csv_files(dataset_path: Path):
    """Return all CSV files in the dataset folder."""
    return sorted(dataset_path.glob("*.csv"))


def main():
    dataset_path = get_dataset_path()
    csv_files = list_csv_files(dataset_path)

    print(f"Dataset Path : {dataset_path}")
    print(f"CSV Files Found : {len(csv_files)}")

    for file in csv_files:
        print(file.name)


if __name__ == "__main__":
    main()