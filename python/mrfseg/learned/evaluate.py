"""Measure a way of making affinities with a simulated user, on whole scans of the cache.

    python -m mrfseg.learned.evaluate --cache /path/to/pseg3d/data/casx/cache [--model runs/walker/best.pt] [--basic]

For each test scan a lumen voxel is drawn (the same one every run; `targets.random_seed`) and
the scan is segmented exactly as the editor would segment it -- as an image of its own, through
`MRFSegmentation`, the whole scan at once -- from that one click; the target is the coronary
tree the click is in (`targets.tree_of`). Then a simulated user (`clicks.py`) clicks up to
`--clicks` times toward the target, inside where lumen is missed and outside where something
else was taken. Reported per way of making affinities: mean Dice after 1, 3, 5 and 10 clicks,
the mean number of clicks to reach Dice 0.85 and 0.90 (never reaching counts as clicks + 1),
how much of the *other* tree the first click's segmentation took in (leak), how many connected
pieces it had (one, when the solve converged), and the seconds per click. `--model` measures a
trained network, `--basic` the contrast affinities (`--set name=value` changes settings:
`--set contrast=60`); with neither, `--basic`. `--out` gets metrics.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from ..extractors import EXTRACTORS, AffinityMaker
from ..grid import image_from_voxels, voxels
from ..segmentation import MRFSegmentation
from ..settings import MRFSettings
from .clicks import clicks_to_reach, simulate
from .data import Case, load_cases
from .extractor import Learned
from .model import load
from .targets import random_seed, tree_of


def scan_target(case: Case, seed: int) -> dict | None:
    """The simulated user's task on `case`: the voxel of the first click (drawn from `seed`),
    the tree it selects, and the other lumen (whose segmentation is a leak). None for a scan
    without lumen."""
    rng = np.random.default_rng([seed, int(case.id) if case.id.isdigit() else abs(hash(case.id)) % 2**31])
    label = np.asarray(case.label)
    at = random_seed(label, rng)
    if at is None:
        return None
    tree = tree_of(label, at)
    return dict(case=case.id, at=at, tree=tree, others=(label > 0) & ~tree, spacing=case.spacing)


def segmenter_for(case: Case, settings: MRFSettings, extractor=None, affinities: AffinityMaker | None = None) -> tuple:
    """clicks (n, 4) rows of (z, y, x, label) -> the labelmap of the scan segmented as the editor
    would: the scan as an image at its spacing, the clicks as points in its world frame; and the
    segmentation itself, for its last inference summary. With an `extractor` (and the
    `affinities` maker that goes with it), that one instead of the settings'."""
    spacing = case.spacing
    image = image_from_voxels(np.asarray(case.image, dtype=np.int16), spacing)
    segmentation = MRFSegmentation(image, settings, extractor, affinities)

    def segment(clicks: np.ndarray) -> np.ndarray:
        clicks = np.asarray(clicks).reshape(-1, 4)
        points = clicks[:, [2, 1, 0]].astype(np.float64) * spacing  # (x, y, z) mm
        result = segmentation.generate_segmentation(points, clicks[:, 3])
        return voxels(result.probability) >= 0.5

    return segment, segmentation


def evaluate_scans(cases: list[Case], settings: MRFSettings, n_clicks: int, extractor=None, affinities: AffinityMaker | None = None,
                   seed: int = 0, log=print) -> dict:
    """The simulated user on every scan: the metrics in the module docstring."""
    dices, leaks, pieces, seconds, iterations = [], [], [], [], []
    for n, case in enumerate(cases):
        target = scan_target(case, seed)
        if target is None:
            continue
        segment, segmentation = segmenter_for(case, settings, extractor, affinities)
        summaries = []

        def recorded(clicks: np.ndarray) -> np.ndarray:
            prediction = segment(clicks)
            summaries.append(segmentation.last_segmentation.inference)
            return prediction

        started = time.perf_counter()
        result = simulate(recorded, target["tree"], n_clicks, others=target["others"], first=target["at"])
        elapsed = time.perf_counter() - started
        first = summaries[0] if summaries else None
        dices.append(result["dice"])
        leaks.append(result["leak"])
        pieces.append(first.components if first else math.nan)
        iterations.append(first.iterations if first else math.nan)
        seconds.append(elapsed / max(len(result["clicks"]), 1))
        log(f"  scan {n + 1}/{len(cases)} ({case.id}): Dice after 1 / 3 / 5 clicks "
            f"{result['dice'][0]:.3f} / {result['dice'][min(2, n_clicks - 1)]:.3f} / {result['dice'][min(4, n_clicks - 1)]:.3f}, "
            f"leak {result['leak'][0]:.3f}, pieces {pieces[-1]}, {iterations[-1]} iterations, {seconds[-1]:.1f} s per click")
    if not dices:
        return dict(scans=0)
    d = np.array(dices)
    mean = d.mean(0)
    return dict(
        scans=len(dices),
        dice={str(k): float(mean[k - 1]) for k in (1, 3, 5, 10) if k <= n_clicks},
        clicks_to_085=float(np.mean([clicks_to_reach(row, 0.85) for row in d])),
        clicks_to_090=float(np.mean([clicks_to_reach(row, 0.90) for row in d])),
        leak1=float(np.mean([row[0] for row in leaks])),
        leak_last=float(np.mean([row[-1] for row in leaks])),
        components1=float(np.nanmean(pieces)) if pieces else math.nan,
        iterations=float(np.nanmean(iterations)) if iterations else math.nan,
        seconds_per_click=float(np.mean(seconds)),
    )


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set wants name=value, not {item!r}")
        if value.lower() in ("true", "false"):
            out[name] = value.lower() == "true"
        elif value.lstrip("-").isdigit():
            out[name] = int(value)
        else:
            try:
                out[name] = float(value)
            except ValueError:
                out[name] = value
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="the 0.5 mm cache (absolute path, e.g. pseg3d's data/casx/cache)")
    ap.add_argument("--model", help="a trained best.pt: measure the learned affinities")
    ap.add_argument("--basic", action="store_true", help="measure the contrast affinities (the default with no --model)")
    ap.add_argument("--set", action="append", default=[], help="a setting, name=value (applies to both)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cases", type=int, default=20, help="scans of the split to use")
    ap.add_argument("--clicks", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="where metrics.json goes")
    args = ap.parse_args(argv)

    cases = load_cases(args.cache, args.split, args.cases)
    print(f"{len(cases)} {args.split} scans, {args.clicks} clicks each", flush=True)
    metrics = {}
    if args.basic or not args.model:
        settings = MRFSettings(feature_extractor="basic", device=args.device, **parse_set(args.set))
        print("basic (contrast affinities):", {k: v for k, v in dataclasses.asdict(settings).items() if v is not None}, flush=True)
        metrics["basic"] = evaluate_scans(cases, settings, args.clicks, seed=args.seed, log=lambda s: print(s, flush=True))
        print(json.dumps(metrics["basic"], indent=1), flush=True)
    if args.model:
        model, extra = load(args.model, "cpu")
        extractor = Learned(model, extra)
        settings = MRFSettings(feature_extractor="learned", device=args.device, **{**extractor.setting_defaults, **parse_set(args.set)})
        print(f"learned ({args.model}):", {**extractor.setting_defaults, **parse_set(args.set)}, flush=True)
        metrics["learned"] = evaluate_scans(cases, settings, args.clicks, extractor, EXTRACTORS["learned"].affinities, args.seed, log=lambda s: print(s, flush=True))
        print(json.dumps(metrics["learned"], indent=1), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(Path(args.out) / "metrics.json", "w") as f:
            json.dump(dict(args=vars(args), metrics=metrics), f, indent=1)


if __name__ == "__main__":
    main()
