from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

import run_phase7_full_pipeline as base

KAGGLE_HANDLE = "alexanderliao/artbench10"


def _install_dataframe_style_column_compat() -> None:
    """Resolve the pandas DataFrame.style namespace collision inside the legacy Phase VII driver.

    The Phase VII driver predates pandas' Styler collision becoming visible in this code path and
    accesses a column named ``style`` via attribute syntax (e.g. ``manifest.style``). In pandas,
    ``DataFrame.style`` is a built-in property returning a Styler object, so attribute access does
    not resolve to the column. For this launcher process only, return the ``style`` Series whenever
    such a column exists; preserve the normal Styler property for all other DataFrames.

    Downstream standalone scripts are fixed separately to use bracket notation explicitly.
    """
    original_style_property = pd.DataFrame.style

    def _style_or_column(self):
        if "style" in self.columns:
            return self["style"]
        return original_style_property.__get__(self, type(self))

    pd.DataFrame.style = property(_style_or_column)
    print("Pandas compatibility patch ✓ DataFrame.style resolves the 'style' column in Phase VII")


def hf256_resources(cache_dir: Path, data_dir: Path, extract_dir: Path):
    """
    Phase-VII acquisition override.

    The official ArtBench server currently returns HTTP 403 to Colab and the Kaggle mirror exposes
    the metadata plus 32x32 distributions, but not the 256x256 ImageFolder. We therefore use:
      - ArtBench-10.csv from the Kaggle mirror for artist/style/split metadata;
      - zguo0525/ArtBench on Hugging Face for the full 60,000 256x256 images.

    prepare_artbench_hf256.py validates the filenames embedded directly in the Hugging Face Image
    field against ArtBench-10.csv within the same split and style. Metadata-linked rows are used for
    artist-dependent analyses; the tiny unmatched remainder is retained only for style/corpus-level
    analyses. No uncertain artist identity is imputed.
    """
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
        raise ImportError("kagglehub is required for ArtBench metadata") from exc

    print("Attaching Kaggle ArtBench mirror for metadata:", KAGGLE_HANDLE)
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
    print("Preparing full 256px ArtBench from Hugging Face with direct filename audit...")
    subprocess.run(cmd, check=True)

    image_root = base.locate_imagefolder_root(hf_root)
    print("Validated HF 256px ArtBench ready ✓", image_root)
    return image_root, metadata, "huggingface_zguo0525_256_direct_filename_validated"


def main(args):
    _install_dataframe_style_column_compat()
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
