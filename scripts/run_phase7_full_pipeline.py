from __future__ import annotations

import argparse
import json
import os
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

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://github.com/liaopeiyuan/artbench",
}


def download_stream(url: str, destination: Path, *, expected_min_bytes: int = 1, allow_resume: bool = True, timeout: int = 120) -> Path:
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
            raise RuntimeError(f"HTTP {r.status_code} while downloading {url}. Response preview: {preview!r}")
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


def obtain_artbench_tar(cache_dir: Path, data_dir: Path) -> Path:
    cache_tar = cache_dir / TAR_NAME
    if cache_tar.exists() and cache_tar.stat().st_size > 1_000_000_000:
        print("Using cached ArtBench tar from Drive ✓", cache_tar)
        return cache_tar
    try:
        import kagglehub
        handle = "alexanderliao/artbench10"
        candidates = [TAR_NAME, f"256X256/{TAR_NAME}", f"data/256X256/{TAR_NAME}", f"ArtBench-10/data/256X256/{TAR_NAME}"]
        for candidate in candidates:
            try:
                print("Trying Kaggle single file:", candidate)
                p = Path(kagglehub.dataset_download(handle, path=candidate))
                hits = [p] if p.is_file() and p.name == TAR_NAME else (list(p.rglob(TAR_NAME)) if p.exists() else [])
                if hits:
                    src = hits[0]
                    if src.stat().st_size > 1_000_000_000:
                        shutil.copy2(src, cache_tar)
                        print("Kaggle single-file download ✓", src)
                        return cache_tar
            except Exception as exc:
                print("  Kaggle candidate failed:", type(exc).__name__, str(exc)[:160])
    except Exception as exc:
        print("KaggleHub unavailable:", type(exc).__name__, exc)
    print("Kaggle single-file path unavailable; using official ArtBench host with browser headers.")
    return download_stream(ARTBENCH_TAR_URL, cache_tar, expected_min_bytes=1_000_000_000, allow_resume=True)


def obtain_metadata(cache_dir: Path) -> Path:
    dest = cache_dir / "ArtBench-10.csv"
    if dest.exists() and dest.stat().st_size > 10_000:
        print("Using cached metadata ✓", dest)
        return dest
    return download_stream(ARTBENCH_METADATA_URL, dest, expected_min_bytes=10_000, allow_resume=False)


def locate_imagefolder_root(root: Path) -> Path:
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root
    candidates = []
    for tr in root.rglob("train"):
        if tr.is_dir() and (tr.parent / "test").is_dir():
            n_classes = sum(x.is_dir() for x in tr.iterdir())
            if n_classes >= 8:
                candidates.append(tr.parent)
    if not candidates:
        raise FileNotFoundError(f"Could not find ArtBench train/test below {root}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def extract_tar_if_needed(tar_path: Path, extract_dir: Path) -> Path:
    try:
        return locate_imagefolder_root(extract_dir)
    except FileNotFoundError:
        pass
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting ArtBench archive...")
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(extract_dir, filter="data")
        except TypeError:
            tf.extractall(extract_dir)
    root = locate_imagefolder_root(extract_dir)
    print("ImageFolder ready ✓", root)
    return root


def key_of(a, b, c) -> str:
    return f"{a}||{b}||{c}"


def main(args):
    repo_dir = Path(args.repo_dir).resolve()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from src.baselines import lbp_features, multidistance_glcm_features, multiscale_gradient_features, orientation_histogram_features
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

    commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    print("Repository commit:", commit)
    print("Persistent root:", drive_root)

    tar_path = obtain_artbench_tar(cache_dir, data_dir)
    image_root = extract_tar_if_needed(tar_path, extract_dir)
    metadata_csv = obtain_metadata(cache_dir)

    manifest_path = results_dir / "artbench_full_manifest.csv"
    subprocess.run([sys.executable, "-u", "scripts/prepare_artbench_manifest.py", "--dataset-root", str(image_root), "--metadata-csv", str(metadata_csv), "--output", str(manifest_path)], check=True)
    manifest = pd.read_csv(manifest_path)
    print("Manifest shape:", manifest.shape)
    print(manifest.groupby(["split", "style"]).size().unstack(fill_value=0))
    if len(manifest) != 60000:
        raise RuntimeError(f"Expected 60000 ArtBench rows, found {len(manifest)}")
    if "metadata_match" in manifest.columns:
        print("Metadata coverage:", float(manifest["metadata_match"].mean()))

    def extract_frozen(path, long_side=256, sigma_refs=(1.0, 2.0, 4.0, 8.0), reference_long_side=512):
        _, I = preprocess(Path(path), long_side=long_side)
        geom = relative_scale_curvature_features(I, long_side=long_side, sigma_refs=sigma_refs, reference_long_side=reference_long_side, return_maps=False)
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
                done.update(key_of(a, b, c) for a, b, c in zip(d.split, d.style, d.filename))
            except Exception as exc:
                print("Checkpoint warning:", p, exc)
        return done

    done = done_keys()
    mask_pending = [key_of(a, b, c) not in done for a, b, c in zip(manifest.split, manifest.style, manifest.filename)]
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

    for rec in tqdm(pending.itertuples(index=False), total=len(pending), desc="Full B90+G44 extraction", dynamic_ncols=True):
        meta = {"split": rec.split, "style": rec.style, "artist": getattr(rec, "artist", ""), "source": getattr(rec, "source", ""), "filename": rec.filename, "path": rec.path, "long_side": 256}
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
    full = pd.concat(parts, ignore_index=True)
    full["_k"] = [key_of(a, b, c) for a, b, c in zip(full.split, full.style, full.filename)]
    full = full.drop_duplicates("_k").drop(columns="_k")
    full.to_csv(full_features, index=False)
    print("Full frozen feature matrix:", full.shape)
    if len(full) != 60000:
        raise RuntimeError(f"Feature extraction produced {len(full)} rows, not 60000. Inspect {failures_path} and rerun; completed chunks will be reused.")

    phase5 = results_dir / "phase7_full_style_geometry"
    phase5b = results_dir / "phase7_full_source_sensitivity"
    if not (phase5 / "phase5_scale_summary.csv").exists():
        subprocess.run([sys.executable, "-u", "scripts/run_phase5_style_geometry.py", "--features", str(full_features), "--output-dir", str(phase5), "--n-permutations", str(args.n_permutations), "--seed", "42"], check=True)
    if not (phase5b / "phase5b_source_sensitivity_primary.csv").exists():
        subprocess.run([sys.executable, "-u", "scripts/run_phase5b_source_sensitivity.py", "--features", str(full_features), "--output-dir", str(phase5b), "--n-permutations", str(args.n_permutations), "--seed", "42"], check=True)

    enriched = results_dir / "artbench_full_features_with_ordinal.csv"
    if not enriched.exists():
        subprocess.run([sys.executable, "-u", "scripts/extract_tarozo_ordinal_features.py", "--features", str(full_features), "--dataset-root", str(image_root), "--output", str(enriched), "--checkpoint-every", str(args.ordinal_checkpoint_every)], check=True)
    enriched_df = pd.read_csv(enriched)
    if len(enriched_df) != 60000:
        raise RuntimeError(f"Ordinal-enriched matrix has {len(enriched_df)} rows")
    if sum(c.startswith("ord75__") for c in enriched_df.columns) != 75:
        raise RuntimeError("Expected 75 OP75 columns")

    confirm = results_dir / "phase7_confirmatory_linear_probes"
    if not (confirm / "phase7_confirmatory_deltas_all.csv").exists():
        subprocess.run([sys.executable, "-u", "scripts/run_phase7_full_confirmatory.py", "--features", str(enriched), "--output-dir", str(confirm), "--outer-folds", "5", "--inner-folds", "3", "--n-boot", str(args.n_bootstrap)], check=True)

    run_manifest = {"repo_commit": commit, "n_manifest": int(len(manifest)), "full_features_rows": int(len(full)), "enriched_rows": int(len(enriched_df)), "n_permutations": int(args.n_permutations), "n_bootstrap": int(args.n_bootstrap), "feature_chunk_size": int(args.feature_chunk_size), "ordinal_checkpoint_every": int(args.ordinal_checkpoint_every), "artbench_tar_url": ARTBENCH_TAR_URL}
    (results_dir / "PHASE7_RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    light = drive_root / "painting_geometry_phase7_full_results_LIGHT.zip"
    features_zip = drive_root / "painting_geometry_phase7_feature_matrices.zip"
    with zipfile.ZipFile(light, "w", zipfile.ZIP_DEFLATED) as z:
        for p in results_dir.rglob("*"):
            if not p.is_file() or p in {full_features, enriched} or "ordinal_checkpoint" in p.name:
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
    p.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/painting_geometry_phase7_full"))
    p.add_argument("--feature-chunk-size", type=int, default=500)
    p.add_argument("--ordinal-checkpoint-every", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=4999)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    args = p.parse_args()
    main(args)
