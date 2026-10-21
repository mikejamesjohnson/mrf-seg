"""Resample CTA volumes and their coronary label maps to the training grid, once, into a cache
the trainer draws patches from.

    python -m mrfseg.learned.prepare --images IMAGES --labels LABELS --out CACHE \
        [--filelist FILELIST] [--centerlines CENTERLINES] [--workers 8]

For ImageCAS-X: IMAGES holds the ImageCAS volumes (`<id>.img.nii.gz`, in subfolders or not --
the Kaggle archives unpack to `1-200/`, `201-400/`, ...), LABELS the label maps on the same
scans (`--label-suffix`: `.nii.gz` for the binary `truths_binary/<id>.nii.gz`, `.coronary.nii.gz`
for segment maps with 0 background and 1-14 coronary segments), FILELIST its `filelist/`
(`train.txt`, `val.txt`, `test.txt`, `exclude.txt`, one scan id per line) and CENTERLINES its
`centerlines/` (`<id>.coronary_{left,right}_centerline.vtk`). Only scans with both an image and
a label map are taken; without a filelist they are split 80/10/10.

Each scan becomes, in CACHE:

    <id>.image.npy       int16 HU, (z, y, x), on a `--spacing` mm isotropic grid
    <id>.label.npy       uint8, the label map on the same grid (labels kept, not binarised)
    <id>.lumen.npy       int32 (n, 3): the (z, y, x) indices of every labelled voxel
    <id>.json            the grid (shape, spacing, origin, direction) and a few counts
    <id>.centerline.npz  with --centerlines: points (n, 3) in LPS mm, with their branch and side

plus `splits.json` and `index.json` (every `<id>.json`, in one file). The image is resampled
linearly and the label map by nearest neighbour onto the image's new grid (SimpleITK, as
ImageCAS-X's own preprocessing does), so the two agree voxel for voxel whatever their headers
said. Scans already in the cache are skipped unless `--overwrite`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import SPACING

IMAGE_SUFFIX = ".img.nii.gz"
LABEL_SUFFIX = ".coronary.nii.gz"
CENTERLINE_SUFFIXES = {"left": ".coronary_left_centerline.vtk", "right": ".coronary_right_centerline.vtk"}
SURFACE_SUFFIX = ".coronary_surface.vtk"


# ------------------------------------------------------------------ the cache's format

def write_case(out: Path, case_id: str, image: np.ndarray, label: np.ndarray, spacing: float,
               origin: tuple[float, float, float], direction: np.ndarray, **extra) -> dict:
    """Write one scan to the cache: `image` (z, y, x) HU and `label` (z, y, x) on a `spacing` mm
    isotropic grid whose voxel (0, 0, 0) is at `origin` (x, y, z, LPS mm) with `direction`'s
    columns as its axes. Returns what `<id>.json` holds."""
    if image.shape != label.shape:
        raise ValueError(f"{case_id}: image {image.shape} and label {label.shape} differ")
    out.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.rint(image), -32768, 32767).astype(np.int16) if image.dtype != np.int16 else image
    label = label.astype(np.uint8)
    lumen = np.argwhere(label > 0).astype(np.int32)
    np.save(out / f"{case_id}.image.npy", np.ascontiguousarray(image))
    np.save(out / f"{case_id}.label.npy", np.ascontiguousarray(label))
    np.save(out / f"{case_id}.lumen.npy", lumen)
    labels, counts = np.unique(label, return_counts=True)
    meta = dict(
        id=case_id,
        shape=list(image.shape),
        spacing=float(spacing),
        origin=[float(v) for v in origin],
        direction=[float(v) for v in np.asarray(direction, dtype=float).ravel()],
        lumen_voxels=int(len(lumen)),
        labels={int(l): int(c) for l, c in zip(labels, counts) if l > 0},
        hu_range=[int(image.min()), int(image.max())],
        **extra,
    )
    with open(out / f"{case_id}.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def read_index(cache: Path) -> dict:
    """`index.json`, or the same assembled from the `<id>.json` files."""
    path = cache / "index.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return assemble_index(cache)


def assemble_index(cache: Path) -> dict:
    cases = {}
    for sidecar in sorted(cache.glob("*.json")):
        if sidecar.name in ("index.json", "splits.json"):
            continue
        with open(sidecar) as f:
            meta = json.load(f)
        if "shape" in meta and (cache / f"{meta['id']}.image.npy").exists():
            cases[meta["id"]] = meta
    spacings = {meta["spacing"] for meta in cases.values()}
    index = dict(spacing=spacings.pop() if len(spacings) == 1 else None, cases=cases)
    with open(cache / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    return index


def read_splits(cache: Path) -> dict[str, list[str]]:
    with open(cache / "splits.json") as f:
        return json.load(f)


# ------------------------------------------------------------------ resampling one scan

def resample_pair(image_path: Path, label_path: Path, spacing: float):
    """The scan and its label map on a `spacing` mm isotropic grid: (image (z, y, x) float32,
    label (z, y, x) uint8, origin, direction (3, 3), original spacing, original size)."""
    import SimpleITK as sitk

    def read(path: Path):
        # `<id>.img.nii.gz` names could pass for an Analyze pair: say what the file is.
        nifti = path.name.endswith((".nii", ".nii.gz"))
        return sitk.ReadImage(str(path), imageIO="NiftiImageIO") if nifti else sitk.ReadImage(str(path))

    image, label = read(image_path), read(label_path)
    size, old = image.GetSize(), image.GetSpacing()
    new_size = [max(1, int(round(n * s / spacing))) for n, s in zip(size, old)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))  # beyond the scan: as dark as it gets
    image_r = resampler.Execute(sitk.Cast(image, sitk.sitkFloat32))
    # The label onto the image's new grid, not its own: the two then agree voxel for voxel
    # even if their headers did not quite.
    labeller = sitk.ResampleImageFilter()
    labeller.SetReferenceImage(image_r)
    labeller.SetInterpolator(sitk.sitkNearestNeighbor)
    labeller.SetDefaultPixelValue(0)
    label_r = labeller.Execute(label)
    direction = np.array(image_r.GetDirection(), dtype=float).reshape(3, 3)
    return (sitk.GetArrayFromImage(image_r), sitk.GetArrayFromImage(label_r).astype(np.uint8),
            tuple(image_r.GetOrigin()), direction, tuple(old), tuple(size))


def read_centerlines(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """The centerline polydata files as arrays: `points` (n, 3) in the files' coordinates (LPS mm
    for ImageCAS-X), `branch` (n,) the index of the polyline each point is on, `side` (n,) the
    index of the file (0 left, 1 right), and every point-data array the files carry."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader

    points, branch, side, arrays = [], [], [], {}
    n_branches = 0
    for s, (name, path) in enumerate(paths.items()):
        if not path.exists():
            continue
        reader = vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.ReadAllScalarsOn()
        reader.ReadAllFieldsOn()
        reader.Update()
        poly = reader.GetOutput()
        if poly.GetNumberOfPoints() == 0:
            continue
        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        branch_of = np.full(len(pts), -1, dtype=np.int32)
        lines, ids = poly.GetLines(), vtkIdList()
        lines.InitTraversal()
        while lines.GetNextCell(ids):
            branch_of[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]] = n_branches
            n_branches += 1
        points.append(pts)
        branch.append(branch_of)
        side.append(np.full(len(pts), s, dtype=np.int32))
        data = poly.GetPointData()
        for i in range(data.GetNumberOfArrays()):
            array = data.GetArray(i)
            if array is not None:
                arrays.setdefault(array.GetName(), []).append(vtk_to_numpy(array))
    if not points:
        return {}
    out = dict(points=np.concatenate(points), branch=np.concatenate(branch), side=np.concatenate(side))
    for name, chunks in arrays.items():
        if sum(len(c) for c in chunks) == len(out["points"]):
            out[name] = np.concatenate(chunks)
    return out


def prepare_case(job: tuple) -> tuple[str, bool, str]:
    """Resample one scan into the cache: (id, ok, message)."""
    case_id, image_path, label_path, centerline_paths, out, spacing = job
    try:
        image, label, origin, direction, old_spacing, old_size = resample_pair(Path(image_path), Path(label_path), spacing)
        meta = write_case(Path(out), case_id, image, label, spacing, origin, direction,
                          source=dict(image=str(image_path), label=str(label_path), spacing=list(old_spacing), size=list(old_size)))
        if centerline_paths:
            arrays = read_centerlines({k: Path(v) for k, v in centerline_paths.items()})
            if arrays:
                np.savez(Path(out) / f"{case_id}.centerline.npz", **arrays)
        return case_id, True, f"{tuple(meta['shape'])} voxels, {meta['lumen_voxels']} lumen"
    except Exception as error:  # noqa: BLE001 -- one bad scan must not stop the rest
        return case_id, False, f"{type(error).__name__}: {error}"


def _worker_init() -> None:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


# ------------------------------------------------------------------ the scans and their splits

def index_files(root: Path, suffix: str) -> dict[str, Path]:
    """Every `<id><suffix>` under `root`, at any depth: id -> path."""
    found = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if path.is_file():
            found[path.name[: -len(suffix)]] = path
    return found


def read_filelist(filelist: Path) -> dict[str, list[str]]:
    """`train.txt`, `val.txt`, `test.txt` as lists of ids, less `exclude.txt`."""
    def ids(name: str) -> list[str]:
        path = filelist / f"{name}.txt"
        return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    exclude = set(ids("exclude"))
    return {split: [i for i in ids(split) if i not in exclude] for split in ("train", "val", "test")}


def default_splits(case_ids: list[str], seed: int = 0) -> dict[str, list[str]]:
    """80 / 10 / 10 by a seeded shuffle."""
    order = list(case_ids)
    np.random.default_rng(seed).shuffle(order)
    n = len(order)
    n_val = max(1, n // 10) if n >= 3 else 0
    n_test = n_val
    return dict(train=sorted(order[: n - n_val - n_test]), val=sorted(order[n - n_val - n_test: n - n_test]), test=sorted(order[n - n_test:]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of CTA volumes (NIfTI or anything SimpleITK reads)")
    ap.add_argument("--labels", required=True, help="directory of label maps on the same scans")
    ap.add_argument("--out", required=True, help="the cache to write")
    ap.add_argument("--filelist", help="directory with train.txt / val.txt / test.txt (/ exclude.txt)")
    ap.add_argument("--centerlines", help="directory of centerline .vtk files, kept with each scan for evaluation")
    ap.add_argument("--spacing", type=float, default=SPACING)
    ap.add_argument("--image-suffix", default=IMAGE_SUFFIX, help="what follows the scan id in an image's file name")
    ap.add_argument("--label-suffix", default=LABEL_SUFFIX, help="likewise for a label map (.nii.gz for truths_binary/)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N scans (for a trial)")
    args = ap.parse_args(argv)

    images, labels, out = Path(args.images), Path(args.labels), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Files may sit in subfolders (the Kaggle archives unpack to 1-200/, 201-400/, ...).
    image_files = index_files(images, args.image_suffix)
    label_files = index_files(labels, args.label_suffix)
    splits = read_filelist(Path(args.filelist)) if args.filelist else None
    if splits:
        wanted = [i for split in ("train", "val", "test") for i in splits[split]]
    else:
        wanted = sorted(label_files, key=lambda i: (len(i), i))
    missing, jobs, skipped = [], [], 0
    for case_id in wanted[: args.limit]:
        image_path, label_path = image_files.get(case_id), label_files.get(case_id)
        if image_path is None or label_path is None:
            missing.append(case_id)
            continue
        if not args.overwrite and (out / f"{case_id}.json").exists():
            skipped += 1
            continue
        centerlines = {side: str(Path(args.centerlines) / f"{case_id}{suffix}") for side, suffix in CENTERLINE_SUFFIXES.items()} if args.centerlines else None
        jobs.append((case_id, str(image_path), str(label_path), centerlines, str(out), args.spacing))
    print(f"{len(wanted)} scans: {len(jobs)} to resample, {skipped} already cached, {len(missing)} without both files", flush=True)
    if missing:
        print("  missing:", " ".join(missing[:20]), "..." if len(missing) > 20 else "", flush=True)

    started = time.perf_counter()
    failed = []
    if jobs:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            futures = {pool.submit(prepare_case, job): job[0] for job in jobs}
            for n, future in enumerate(as_completed(futures), 1):
                case_id, ok, message = future.result()
                print(f"  [{n}/{len(jobs)}] {case_id}: {'ok' if ok else 'FAILED'} {message}", flush=True)
                if not ok:
                    failed.append(case_id)

    index = assemble_index(out)
    present = set(index["cases"])
    if splits:
        splits = {split: [i for i in ids if i in present] for split, ids in splits.items()}
    else:
        splits = default_splits(sorted(present))
    with open(out / "splits.json", "w") as f:
        json.dump(splits, f, indent=1)
    print(f"\ncache {out}: {len(present)} scans at {args.spacing:g} mm, "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])} train / val / test, "
          f"{time.perf_counter() - started:.0f} s" + (f"; {len(failed)} failed: {' '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
