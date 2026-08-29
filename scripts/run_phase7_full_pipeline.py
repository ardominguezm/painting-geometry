from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

ARTBENCH_TAR_URL = "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar"
ARTBENCH_METADATA_URL = "https://artbench.eecs.berkeley.edu/files/ArtBench-10.csv"
TAR_NAME = "artbench-10-imagefolder-split.tar"
KAGGLE_HANDLE = "alexanderliao/artbench10"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://github.com/liaopeiyuan/artbench",
}


def norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def download_stream(
    url: str,
    destination: Path,
    *,
    expected_min_bytes: int = 1,
    allow_resume: bool = True,
    timeout: int = 120,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    start = part.stat().st_size if (allow_resume and part.exists()) else 0
    headers = dict(BROWSER_HEADERS)
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        print(f"Resuming {destination.name} from {start/1e6:.1f} MB")
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=timeout) as r:
        if start > 0 and r.status_code == 200:
            print("Server ignored HTTP Range; restarting download.")
            start = 0
            part.unlink(missing_ok=True)
        elif r.status_code not in (200, 206):
            preview = ""
            try:
                preview = r.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"HTTP {r.status_code} while downloading {url}. "
                f"Response preview: {preview!r}"
            )
        mode = "ab" if (start > 0 and r.status_code == 206) else "wb"
        total_hdr = r.headers.get("content-length")
        total = int(total_hdr) + start if total_hdr and total_hdr.isdigit() else None
        downloaded = start
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100 * 1024 * 1024) < len(chunk):
                    if total:
                        print(f"  {downloaded/1e9:.2f}/{total/1e9:.2f} GB")
                    else:
                        print(f"  {downloaded/1e9:.2f} GB")
    size = part.stat().st_size
    if size < expected_min_bytes:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {size} bytes ({part})")
    part.replace(destination)
    print(f"Downloaded ✓ {destination} ({destination.stat().st_size/1e9:.2f} GB)")
    return destination


def imagefolder_candidates(root: Path) -> list[tuple[int, int, int, Path]]:
    """Return plausible train/test ImageFolder roots, scored for the 60k 256px ArtBench split."""
    roots: list[tuple[int, int, int, Path]] = []
    if not root.exists():
        return roots

    possible = []
    if (root / "train").is_dir() and (root / "test").is_dir():
        possible.append(root)
    for tr in root.rglob("train"):
        if tr.is_dir() and (tr.parent / "test").is_dir():
            possible.append(tr.parent)

    seen = set()
    for p in possible:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        train = p / "train"
        test = p / "test"
        style_dirs = [d for d in train.iterdir() if d.is_dir()]
        n_classes = len(style_dirs)
        if n_classes < 8:
            continue
        n_images = 0
        for split_dir in (train, test):
            for q in split_dir.rglob("*"):
                if q.is_file() and q.suffix.lower() in IMAGE_EXTS:
                    n_images += 1
        path_bonus = 1 if "256" in str(p).lower() else 0
        roots.append((path_bonus, n_classes, n_images, p))
    return roots


def locate_imagefolder_root(root: Path) -> Path:
    cands = imagefolder_candidates(root)
    if not cands:
        raise FileNotFoundError(f"Could not find ArtBench train/test below {root}")
    # Prefer a 256-marked path, then 10 classes, then image count closest to 60k.
    cands.sort(
        key=lambda x: (
            x[0],
            1 if x[1] == 10 else 0,
            -abs(x[2] - 60000),
            x[2],
        ),
        reverse=True,
    )
    print("ImageFolder candidates discovered:")
    for bonus, ncls, nimg, p in cands[:8]:
        print(f"  classes={ncls:2d} images={nimg:6d} 256_hint={bool(bonus)} -> {p}")
    best = cands[0][3]
    print("Selected ImageFolder root ✓", best)
    return best


def extract_tar_if_needed(tar_path: Path, extract_dir: Path) -> Path:
    try:
        return locate_imagefolder_root(extract_dir)
    except FileNotFoundError:
        pass
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting ArtBench archive:", tar_path)
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(extract_dir, filter="data")
        except TypeError:
            tf.extractall(extract_dir)
    root = locate_imagefolder_root(extract_dir)
    print("ImageFolder ready ✓", root)
    return root


def looks_like_artist_metadata(csv_path: Path) -> bool:
    try:
        cols = [norm_token(c) for c in pd.read_csv(csv_path, nrows=3).columns]
    except Exception:
        return False
    has_file = any(c in cols for c in ["filename", "file_name", "image_name", "image", "name", "path"])
    has_artist = any(c in cols for c in ["artist", "artist_name", "creator", "author"])
    return has_file and has_artist


def find_metadata_in_tree(root: Path) -> Path | None:
    if not root.exists():
        return None
    exact_names = ["ArtBench-10.csv", "artbench-10.csv", "artbench10.csv"]
    for name in exact_names:
        hits = list(root.rglob(name))
        for p in hits:
            if looks_like_artist_metadata(p):
                print("Found ArtBench artist metadata in Kaggle dataset ✓", p)
                return p
    for p in root.rglob("*.csv"):
        if looks_like_artist_metadata(p):
            print("Found compatible artist metadata in Kaggle dataset ✓", p)
            return p
    return None


def obtain_metadata(cache_dir: Path, kaggle_root: Path | None = None) -> Path:
    dest = cache_dir / "ArtBench-10.csv"
    if dest.exists() and dest.stat().st_size > 10_000 and looks_like_artist_metadata(dest):
        print("Using cached metadata ✓", dest)
        return dest

    if kaggle_root is not None:
        meta = find_metadata_in_tree(kaggle_root)
        if meta is not None:
            shutil.copy2(meta, dest)
            print("Cached Kaggle metadata to Drive ✓", dest)
            return dest

    print("Artist metadata not found in Kaggle tree; trying official metadata URL...")
    try:
        return download_stream(
            ARTBENCH_METADATA_URL,
            dest,
            expected_min_bytes=10_000,
            allow_resume=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the ArtBench artist metadata required for the artist-disjoint analysis. "
            "The image corpus may be available, but this run deliberately stops rather than silently "
            "dropping the artist-disjoint protocol. Original error: " + repr(exc)
        ) from exc


def obtain_artbench_resources(cache_dir: Path, data_dir: Path, extract_dir: Path) -> tuple[Path, Path, str]:
    """
    Obtain the full 256x256 train/test corpus and artist metadata.

    Priority:
    1) already extracted local runtime;
    2) cached Drive tar + cached metadata;
    3) KaggleHub full-dataset mount/cache (preferred in Colab);
    4) Kaggle-hosted tar found recursively;
    5) official ArtBench host as a final fallback.
    """
    try:
        image_root = locate_imagefolder_root(extract_dir)
        metadata = obtain_metadata(cache_dir, None)
        return image_root, metadata, "local_extracted"
    except Exception:
        pass

    cache_tar = cache_dir / TAR_NAME
    if cache_tar.exists() and cache_tar.stat().st_size > 1_000_000_000:
        print("Using cached ArtBench tar from Drive ✓", cache_tar)
        image_root = extract_tar_if_needed(cache_tar, extract_dir)
        metadata = obtain_metadata(cache_dir, None)
        return image_root, metadata, "drive_cached_tar"

    kaggle_root: Path | None = None
    try:
        import kagglehub

        print("Attaching/downloading the complete Kaggle dataset:", KAGGLE_HANDLE)
        kaggle_root = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
        print("Kaggle dataset root:", kaggle_root)
        if kaggle_root.exists():
            top = sorted(kaggle_root.iterdir(), key=lambda p: p.name.lower())
            print("Kaggle top-level entries (first 30):")
            for p in top[:30]:
                print(" ", p.name, "/" if p.is_dir() else "")

            # Best case: Kaggle exposes the train/test folders already extracted.
            try:
                image_root = locate_imagefolder_root(kaggle_root)
                metadata = obtain_metadata(cache_dir, kaggle_root)
                return image_root, metadata, "kaggle_extracted_imagefolder"
            except FileNotFoundError:
                print("No already-extracted 60k ImageFolder found in Kaggle tree; searching for archive...")

            tar_hits = []
            for p in kaggle_root.rglob("*.tar"):
                name = p.name.lower()
                if "imagefolder" in name and "split" in name:
                    tar_hits.append(p)
            exact = list(kaggle_root.rglob(TAR_NAME))
            for p in exact:
                if p not in tar_hits:
                    tar_hits.insert(0, p)
            tar_hits = [p for p in tar_hits if p.exists() and p.stat().st_size > 1_000_000_000]
            if tar_hits:
                tar_path = sorted(tar_hits, key=lambda p: (p.name != TAR_NAME, len(str(p))))[0]
                print("Found Kaggle ArtBench archive ✓", tar_path, f"({tar_path.stat().st_size/1e9:.2f} GB)")
                image_root = extract_tar_if_needed(tar_path, extract_dir)
                metadata = obtain_metadata(cache_dir, kaggle_root)
                return image_root, metadata, "kaggle_tar"
    except Exception as exc:
        print("Kaggle full-dataset acquisition failed:", type(exc).__name__, str(exc)[:500])

    print("Kaggle did not expose the 256x256 split. Trying official ArtBench host as final fallback...")
    try:
        tar_path = download_stream(
            ARTBENCH_TAR_URL,
            cache_tar,
            expected_min_bytes=1_000_000_000,
            allow_resume=True,
        )
        image_root = extract_tar_if_needed(tar_path, extract_dir)
        metadata = obtain_metadata(cache_dir, kaggle_root)
        return image_root, metadata, "official_host"
    except Exception as exc:
        raise RuntimeError(
            "ArtBench acquisition failed. The official host is currently rejecting Colab with HTTP 403, "
            "and Kaggle did not expose a usable 256x256 train/test ImageFolder or its tar archive. "
            "The diagnostic log above lists the Kaggle dataset root and its top-level entries. "
            "No scientific settings were changed. Original error: " + repr(exc)
        ) from exc


def key_of(a, b, c) -> str:
    return f"{a}||{b}||{c}"


def main(args):
    repo_dir = Path(args.repo_dir).resolve()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from src.baselines import (
        lbp_features,
        multidistance_glcm_features,
        multiscale_gradient_features,
        orientation_histogram_features,
    )
    from src.curvature_v2 import relative_scale_curvature_features
    from src.orientation import structure_tensor_features
    from src.preprocessing import preprocess

    drive_root = Path(args.drive_root)
    cache_dir = drive_root / "cache"
    checkpoint_dir = drive_root / "checkpoints" / "B90_G44_chunks"
    results_dir = drive_root / "results"
    data_dir = Path("/content/artbench_data")
    extract_dir = data_dir / "imagefolder"
    for p in [drive_root, cache_dir, checkpoint_dir, results_dir, data_dir]:
        p.mkdir(parents=True, exist_ok=True)

    commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    print("Repository commit:", commit)
    print("Persistent root:", drive_root)

    image_root, metadata_csv, acquisition_source = obtain_artbench_resources(
        cache_dir, data_dir, extract_dir
    )
    print("ArtBench acquisition source:", acquisition_source)
    print("Image root:", image_root)
    print("Metadata:", metadata_csv)

    manifest_path = results_dir / "artbench_full_manifest.csv"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/prepare_artbench_manifest.py",
            "--dataset-root",
            str(image_root),
            "--metadata-csv",
            str(metadata_csv),
            "--output",
            str(manifest_path),
        ],
        check=True,
    )
    manifest = pd.read_csv(manifest_path)
    print("Manifest shape:", manifest.shape)
    print(manifest.groupby(["split", "style"]).size().unstack(fill_value=0))
    if len(manifest) != 60000:
        raise RuntimeError(f"Expected 60000 ArtBench rows, found {len(manifest)}")
    if "metadata_match" in manifest.columns:
        metadata_coverage = float(manifest["metadata_match"].mean())
        artist_coverage = float(
            manifest["artist"].fillna("").astype(str).str.strip().ne("").mean()
        )
        print("Metadata filename match coverage:", metadata_coverage)
        print("Usable artist coverage:", artist_coverage)
        if artist_coverage < 0.80:
            raise RuntimeError(
                f"Usable artist coverage is only {artist_coverage:.1%}; stopping because artist-disjoint "
                "evaluation is a frozen primary protocol."
            )

    def extract_frozen(
        path,
        long_side=256,
        sigma_refs=(1.0, 2.0, 4.0, 8.0),
        reference_long_side=512,
    ):
        _, I = preprocess(Path(path), long_side=long_side)
        geom = relative_scale_curvature_features(
            I,
            long_side=long_side,
            sigma_refs=sigma_refs,
            reference_long_side=reference_long_side,
            return_maps=False,
        )
        orient_sigma = 2.0 * long_side / reference_long_side
        orient = structure_tensor_features(I, sigma=orient_sigma)
        sigma_px = tuple(s * long_side / reference_long_side for s in sigma_refs)
        base = {}
        base.update(multiscale_gradient_features(I, sigmas=sigma_px))
        base.update(orientation_histogram_features(I, sigma=orient_sigma))
        base.update(multidistance_glcm_features(I, distances=(1, 2, 4)))
        base.update(lbp_features(I))
        out = {f"geom__curv__{k}": v for k, v in geom.items()}
        out.update({f"geom__orient__{k}": v for k, v in orient.items()})
        out.update({f"base__{k}": v for k, v in base.items()})
        return out

    full_features = results_dir / "artbench_full_B90_G44_features.csv"
    failures_path = results_dir / "artbench_full_B90_G44_failures.csv"

    def done_keys():
        done = set()
        for p in sorted(checkpoint_dir.glob("chunk_*.csv")):
            try:
                d = pd.read_csv(p, usecols=["split", "style", "filename"])
                done.update(
                    key_of(a, b, c)
                    for a, b, c in zip(d.split, d.style, d.filename)
                )
            except Exception as exc:
                print("Checkpoint warning:", p, exc)
        return done

    done = done_keys()
    mask_pending = [
        key_of(a, b, c) not in done
        for a, b, c in zip(manifest.split, manifest.style, manifest.filename)
    ]
    pending = manifest[mask_pending].copy()
    print("Frozen B90/G44 extraction — done:", len(done), "remaining:", len(pending))

    chunk_no = len(list(checkpoint_dir.glob("chunk_*.csv")))
    buffer = []
    failures = []

    def flush():
        nonlocal buffer, chunk_no
        if not buffer:
            return
        p = checkpoint_dir / f"chunk_{chunk_no:05d}.csv"
        pd.DataFrame(buffer).to_csv(p, index=False)
        print("CHECKPOINT ✓", p.name, len(buffer))
        buffer = []
        chunk_no += 1

    for rec in tqdm(
        pending.itertuples(index=False),
        total=len(pending),
        desc="Full B90+G44 extraction",
        dynamic_ncols=True,
    ):
        meta = {
            "split": rec.split,
            "style": rec.style,
            "artist": getattr(rec, "artist", ""),
            "source": getattr(rec, "source", ""),
            "filename": rec.filename,
            "path": rec.path,
            "long_side": 256,
        }
        try:
            meta.update(extract_frozen(rec.path))
            buffer.append(meta)
        except Exception as exc:
            failures.append({**meta, "error": repr(exc)})
        if len(buffer) >= args.feature_chunk_size:
            flush()
            if failures:
                pd.DataFrame(failures).to_csv(failures_path, index=False)

    flush()
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)

    parts = [pd.read_csv(p) for p in sorted(checkpoint_dir.glob("chunk_*.csv"))]
    if not parts:
        raise RuntimeError("No feature checkpoint chunks were produced.")
    full = pd.concat(parts, ignore_index=True)
    full["_k"] = [
        key_of(a, b, c) for a, b, c in zip(full.split, full.style, full.filename)
    ]
    full = full.drop_duplicates("_k").drop(columns="_k")
    full.to_csv(full_features, index=False)
    print("Full frozen feature matrix:", full.shape)
    if len(full) != 60000:
        raise RuntimeError(
            f"Feature extraction produced {len(full)} rows, not 60000. "
            f"Inspect {failures_path} and rerun; completed chunks will be reused."
        )

    phase5 = results_dir / "phase7_full_style_geometry"
    phase5b = results_dir / "phase7_full_source_sensitivity"
    if not (phase5 / "phase5_scale_summary.csv").exists():
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/run_phase5_style_geometry.py",
                "--features",
                str(full_features),
                "--output-dir",
                str(phase5),
                "--n-permutations",
                str(args.n_permutations),
                "--seed",
                "42",
            ],
            check=True,
        )
    if not (phase5b / "phase5b_source_sensitivity_primary.csv").exists():
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/run_phase5b_source_sensitivity.py",
                "--features",
                str(full_features),
                "--output-dir",
                str(phase5b),
                "--n-permutations",
                str(args.n_permutations),
                "--seed",
                "42",
            ],
            check=True,
        )

    enriched = results_dir / "artbench_full_features_with_ordinal.csv"
    if not enriched.exists():
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/extract_tarozo_ordinal_features.py",
                "--features",
                str(full_features),
                "--dataset-root",
                str(image_root),
                "--output",
                str(enriched),
                "--checkpoint-every",
                str(args.ordinal_checkpoint_every),
            ],
            check=True,
        )
    enriched_df = pd.read_csv(enriched)
    if len(enriched_df) != 60000:
        raise RuntimeError(f"Ordinal-enriched matrix has {len(enriched_df)} rows")
    if sum(c.startswith("ord75__") for c in enriched_df.columns) != 75:
        raise RuntimeError("Expected 75 OP75 columns")

    confirm = results_dir / "phase7_confirmatory_linear_probes"
    if not (confirm / "phase7_confirmatory_deltas_all.csv").exists():
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/run_phase7_full_confirmatory.py",
                "--features",
                str(enriched),
                "--output-dir",
                str(confirm),
                "--outer-folds",
                "5",
                "--inner-folds",
                "3",
                "--n-boot",
                str(args.n_bootstrap),
            ],
            check=True,
        )

    run_manifest = {
        "repo_commit": commit,
        "n_manifest": int(len(manifest)),
        "full_features_rows": int(len(full)),
        "enriched_rows": int(len(enriched_df)),
        "n_permutations": int(args.n_permutations),
        "n_bootstrap": int(args.n_bootstrap),
        "feature_chunk_size": int(args.feature_chunk_size),
        "ordinal_checkpoint_every": int(args.ordinal_checkpoint_every),
        "artbench_acquisition_source": acquisition_source,
        "artbench_image_root": str(image_root),
        "artbench_metadata": str(metadata_csv),
    }
    (results_dir / "PHASE7_RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    light = drive_root / "painting_geometry_phase7_full_results_LIGHT.zip"
    features_zip = drive_root / "painting_geometry_phase7_feature_matrices.zip"
    with zipfile.ZipFile(light, "w", zipfile.ZIP_DEFLATED) as z:
        for p in results_dir.rglob("*"):
            if (
                not p.is_file()
                or p in {full_features, enriched}
                or "ordinal_checkpoint" in p.name
            ):
                continue
            z.write(p, p.relative_to(results_dir))
    with zipfile.ZipFile(features_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(full_features, full_features.name)
        z.write(enriched, enriched.name)

    print("\nPHASE VII COMPLETE ✓")
    print("LIGHT:", light, f"{light.stat().st_size/1e6:.1f} MB")
    print("FEATURES:", features_zip, f"{features_zip.stat().st_size/1e6:.1f} MB")
    print("Persistent root:", drive_root)


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
    args = p.parse_args()
    main(args)
