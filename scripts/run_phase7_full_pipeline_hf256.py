from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run_phase7_full_pipeline as base

KAGGLE_HANDLE = "alexanderliao/artbench10"


def hf256_resources(cache_dir: Path, data_dir: Path, extract_dir: Path):
    """
    Phase-VII acquisition override.

    The official ArtBench server currently returns HTTP 403 to Colab and the Kaggle mirror
    contains only the CIFAR-style 32x32 distribution. We therefore use:
      - Kaggle ArtBench-10.csv + official 32x32 Python batches for filenames/artist linkage;
      - zguo0525/ArtBench on Hugging Face for the full 60,000 256x256 images.

    prepare_artbench_hf256.py validates the recovered filename ordering against image content
    before any artist metadata are attached. If that validation fails, the run stops.
    """
    # Reuse a materialized 256px tree if the current runtime already has one.
    hf_root = data_dir / "hf256_imagefolder"
    try:
        image_root = base.locate_imagefolder_root(hf_root)
        metadata = base.obtain_metadata(cache_dir, None)
        print("Using already materialized HF 256px ImageFolder ✓", image_root)
        return image_root, metadata, "huggingface_256_materialized_reuse"
    except Exception:
        pass

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("kagglehub is required for ArtBench metadata/32px validation") from exc

    print("Attaching Kaggle ArtBench mirror for metadata + 32px validation:", KAGGLE_HANDLE)
    kaggle_root = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    print("Kaggle dataset root:", kaggle_root)
    if not kaggle_root.exists():
        raise FileNotFoundError(kaggle_root)

    metadata = base.obtain_metadata(cache_dir, kaggle_root)
    audit_csv = cache_dir.parent / "results" / "hf256_filename_recovery_audit.csv"

    cmd = [
        sys.executable,
        "-u",
        "scripts/prepare_artbench_hf256.py",
        "--kaggle-root",
        str(kaggle_root),
        "--output-root",
        str(hf_root),
        "--audit-csv",
        str(audit_csv),
    ]
    print("Preparing validated full 256px ArtBench from Hugging Face...")
    subprocess.run(cmd, check=True)

    image_root = base.locate_imagefolder_root(hf_root)
    print("Validated HF 256px ArtBench ready ✓", image_root)
    return image_root, metadata, "huggingface_zguo0525_256_validated_against_official_32px"


def main(args):
    base.obtain_artbench_resources = hf256_resources
    base.main(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", type=Path, required=True)
    p.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/painting_geometry_phase7_full"),
    )
    p.add_argument("--feature-chunk-size", type=int, default=500)
    p.add_argument("--ordinal-checkpoint-every", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=4999)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    a = p.parse_args()
    main(a)
