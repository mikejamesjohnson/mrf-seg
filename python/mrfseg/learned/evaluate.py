"""Measure a way of making potentials with a simulated user, on the cache's test scans.

    python -m mrfseg.learned.evaluate --cache /path/to/pseg3d/data/casx/cache [--model runs/first/best.pt] [--basic]

For each test scan, `--targets` labelled voxels are drawn (the same ones every run); around each
a `--cube`-voxel sub-volume is cut and segmented exactly as the editor would segment it -- as an
image of its own, through `MRFSegmentation` -- while a simulated user (`clicks.py`) clicks up to
`--clicks` times toward the lumen connected to the drawn voxel within the cube. Reported per way
of making potentials: mean Dice after 1, 3, 5 and 10 clicks, the mean number of clicks to reach
Dice 0.85 and 0.90 (never reaching counts as clicks + 1), and how much of the *other* lumen in
the cube -- other vessels, or the same vessel joined only outside the cube -- the segmentation
took in after 5 clicks (leak). `--model` measures a trained network, `--basic` the handcrafted
potentials (`--set name=value` changes their settings: `--set coupling=3 --set basic_scales=2`);
with neither, `--basic`. `--out` gets metrics.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..extractors import EXTRACTORS, PotentialsMaker
from ..grid import image_from_voxels, voxels
from ..segmentation import MRFSegmentation
from ..settings import MRFSettings
from .clicks import clicks_to_reach, simulate
from .data import SIX, Case, cut, load_cases
from .extractor import Learned
from .model import load


def targets_in(cases: list[Case], per_case: int, cube: int, seed: int) -> list[dict]:
    """Sub-volumes about labelled voxels of each case: HU, the target lumen (connected to the
    voxel within the cube), the other lumen, the voxel itself and the segment label there."""
    out = []
    for case in cases:
        rng = np.random.default_rng([seed, int(case.id) if case.id.isdigit() else abs(hash(case.id)) % 2**31])
        if not len(case.lumen):
            continue
        for _ in range(per_case):
            centre = case.lumen[rng.integers(len(case.lumen))]
            origin = np.clip(centre - cube // 2, 0, np.maximum(np.array(case.shape) - cube, 0))
            label = cut(case.label, origin, cube, fill=0)
            lumen = label > 0
            components, _ = ndimage.label(lumen, structure=SIX)
            at = tuple(int(v) for v in np.clip(centre - origin, 0, cube - 1))
            target = components == components[at] if components[at] else lumen
            hu = cut(case.image, origin, cube, fill=int(case.meta.get("hu_range", [-1024])[0])).astype(np.float32)
            out.append(dict(case=case.id, origin=origin.tolist(), hu=hu, target=target, others=lumen & ~target,
                            at=at, segment=int(label[at]), spacing=case.spacing))
    return out


def segmenter_for(cube: dict, settings: MRFSettings, extractor=None, potentials: PotentialsMaker | None = None):
    """clicks (n, 4) rows of (z, y, x, label) -> the labelmap of the cube segmented as the editor
    would: the cube as an image at its spacing, the clicks as points in its world frame. With an
    `extractor` (and the `potentials` maker that goes with it), that one instead of the settings'."""
    spacing = float(cube["spacing"])
    image = image_from_voxels(cube["hu"].astype(np.int16), spacing)
    segmentation = MRFSegmentation(image, settings, extractor, potentials)

    def segment(clicks: np.ndarray) -> np.ndarray:
        clicks = np.asarray(clicks).reshape(-1, 4)
        points = clicks[:, [2, 1, 0]].astype(np.float64) * spacing  # (x, y, z) mm
        result = segmentation.generate_segmentation(points, clicks[:, 3])
        return voxels(result.probability) >= 0.5

    return segment


def evaluate(cubes: list[dict], settings: MRFSettings, n_clicks: int, extractor=None, potentials: PotentialsMaker | None = None, log=print) -> dict:
    """The simulated user on every cube: the metrics above."""
    dices, leaks, seconds = [], [], []
    for n, cube in enumerate(cubes):
        started = time.perf_counter()
        result = simulate(segmenter_for(cube, settings, extractor, potentials), cube["target"], n_clicks, others=cube["others"], first=cube["at"])
        dices.append(result["dice"])
        leaks.append(result["leak"][min(4, n_clicks - 1)] if result["leak"] else 0.0)
        seconds.append(time.perf_counter() - started)
        log(f"  cube {n + 1}/{len(cubes)} ({cube['case']}, segment {cube['segment']}): Dice after 1 / 3 / 5 clicks "
            f"{result['dice'][0]:.3f} / {result['dice'][min(2, n_clicks - 1)]:.3f} / {result['dice'][min(4, n_clicks - 1)]:.3f}, {seconds[-1]:.1f} s")
    dices = np.array(dices)
    mean = dices.mean(0)
    return dict(
        cubes=len(cubes),
        dice={str(k): float(mean[k - 1]) for k in (1, 3, 5, 10) if k <= n_clicks},
        clicks_to_085=float(np.mean([clicks_to_reach(d, 0.85) for d in dices])),
        clicks_to_090=float(np.mean([clicks_to_reach(d, 0.90) for d in dices])),
        leak5=float(np.mean(leaks)),
        seconds_per_cube=float(np.mean(seconds)),
    )


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set wants name=value, not {item!r}")
        out[name] = {"true": True, "false": False}.get(value.lower(), None) if value.lower() in ("true", "false") else (int(value) if value.lstrip("-").isdigit() else float(value))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="the 0.5 mm cache (absolute path, e.g. pseg3d's data/casx/cache)")
    ap.add_argument("--model", help="a trained best.pt: measure the learned potentials")
    ap.add_argument("--basic", action="store_true", help="measure the handcrafted potentials (the default with no --model)")
    ap.add_argument("--set", action="append", default=[], help="a setting for the handcrafted potentials, name=value")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cases", type=int, default=20, help="scans of the split to use")
    ap.add_argument("--targets", type=int, default=1, help="cubes per scan")
    ap.add_argument("--cube", type=int, default=96, help="voxels a side")
    ap.add_argument("--clicks", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="where metrics.json goes")
    args = ap.parse_args(argv)

    cases = load_cases(args.cache, args.split, args.cases)
    cubes = targets_in(cases, args.targets, args.cube, args.seed)
    print(f"{len(cubes)} cubes of {args.cube}^3 from {len(cases)} {args.split} scans", flush=True)
    metrics = {}
    if args.basic or not args.model:
        settings = MRFSettings(feature_spacing=None, device=args.device, **parse_set(args.set))
        print("basic (handcrafted potentials):", {k: v for k, v in dataclasses.asdict(settings).items() if v is not None}, flush=True)
        metrics["basic"] = evaluate(cubes, settings, args.clicks, log=lambda s: print(s, flush=True))
        print(json.dumps(metrics["basic"], indent=1), flush=True)
    if args.model:
        model, extra = load(args.model, "cpu")
        extractor = Learned(model, extra)
        settings = MRFSettings(device=args.device, **extractor.setting_defaults)
        print(f"learned ({args.model}):", extractor.setting_defaults, flush=True)
        metrics["learned"] = evaluate(cubes, settings, args.clicks, extractor, EXTRACTORS["learned"].potentials, log=lambda s: print(s, flush=True))
        print(json.dumps(metrics["learned"], indent=1), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(Path(args.out) / "metrics.json", "w") as f:
            json.dump(dict(args=vars(args), metrics=metrics), f, indent=1)


if __name__ == "__main__":
    main()
