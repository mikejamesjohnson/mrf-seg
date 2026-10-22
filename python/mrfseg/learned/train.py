"""Train the affinity network on the cache `prepare.py` wrote (or pseg3d's, by absolute path).

    python -m mrfseg.learned.train --cache /path/to/pseg3d/data/casx/cache --out runs/affinity [--steps 40000 --batch 4 ...]
    python -m mrfseg.learned.train --cache ... --out runs/walker --init runs/affinity/best.pt --walker-weight 1 --patch 160 --batch 1

Two stages, one script (`learned/__init__.py`): with `--walker-weight 0` (the default) the loss
is the edge-affinity cross-entropy alone (`affinity_loss`); with a positive weight the walker
is also seeded at each patch's simulated click and solved, and its probability is scored
against the click's tree (`walker_loss`), through the implicit gradient of `rw.RandomWalkerFunction`
-- one more solve per patch, nothing unrolled, so the patch can be large (`--patch 160`).

Every `--log-every` steps it prints the losses and the Dice of the walker on the training
patches (when it runs); every `--val-every` steps the loss on fixed validation patches, and a
simulated user's Dice after 1 / 3 / 5 clicks on `--val-cases` whole validation scans, segmented
exactly as the editor would segment them (`evaluate.evaluate_scans`), with the leak into the
other tree and the number of pieces after the first click. `best.pt` is the checkpoint with the
best single-click Dice, `last.pt` the latest, with the optimiser. A run can be done in pieces:
`--time-limit MINUTES` or `--stop-after STEPS` ends one, validating and saving where it got to,
and `--resume` carries on from there with fresh patches (the data pipeline is seeded by the
step, so a piece never replays the last one's draws) on the schedule `--steps` defines -- keep
that the same across the pieces. On CUDA the network runs in bf16 (Ampere and later) or fp16
with loss scaling (Turing and older); the solver and the losses run in fp32. The defaults --
batch 4, 96^3 patches, bf16 -- are for a 24 GB GPU in the affinity stage; the walker stage
takes `--patch 160 --batch 1` (or 2 on 32 GB).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..extractors import EXTRACTORS
from ..rw import Seeds, SolverOptions, affinities, random_walker_batched
from ..settings import MRFSettings
from .data import PatchConfig, PatchDataset, load_cases
from .evaluate import evaluate_scans
from .extractor import Learned
from .model import AffinityNet, UNet3DConfig, parameter_count, save
from .targets import edge_targets

#: The background seeds in training: the editor's defaults (`settings.MRFSettings`).
SEED_HU = 0.0
SEED_EROSION = 1

# ------------------------------------------------------------------ the losses


def affinity_loss(z: torch.Tensor, label: torch.Tensor, hu: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The edge cross-entropy of a batch: `z` (B, 3, z, y, x) log-affinities from the network,
    `label` (B, z, y, x) 0/1, `hu` (B, z, y, x). The lumen-touching edges and the background
    edges each contribute their mean (`targets.edge_targets`). Returns it and numbers to log:
    the two means, and how often a lumen-touching edge's sign is right."""
    z = z.float()
    targets, touching, exists = edge_targets(label, hu.float())
    exists = exists[None].expand_as(touching)
    bce = F.binary_cross_entropy_with_logits(z, targets, reduction="none")
    lumen_edges, background_edges = touching & exists, ~touching & exists
    zero = z.sum() * 0
    lumen = bce[lumen_edges].mean() if bool(lumen_edges.any()) else zero
    background = bce[background_edges].mean() if bool(background_edges.any()) else zero
    with torch.no_grad():
        right = ((z > 0) == (targets > 0.5))[lumen_edges]
        stats = dict(lumen_bce=float(lumen), background_bce=float(background), edge_accuracy=float(right.float().mean()) if right.numel() else math.nan,
                     lumen_edges=float(lumen_edges.float().mean()))
    return lumen + background, stats


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - the soft Dice of each item, averaged: (B, ...) probabilities against (B, ...) 0/1."""
    p = probability.flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def walker_loss(z: torch.Tensor, hu: torch.Tensor, label: torch.Tensor, seed: torch.Tensor, tree: torch.Tensor,
                options: SolverOptions) -> tuple[torch.Tensor | None, dict]:
    """The walker's loss on the items of a batch that have a seed: the walker solved on the
    affinities exp(z) with the seed (3,) of each item inside and the background below SEED_HU
    outside, its probability scored against the seed's `tree` with cross-entropy and soft Dice.
    None when no item has a seed. The stats: the hard Dice, the mean probability on the tree,
    the leak (mean probability on the lumen that is not the tree) and the solver's iterations."""
    has = seed[:, 0] >= 0
    if not bool(has.any()):
        return None, {}
    index = torch.nonzero(has).ravel()
    with torch.autocast(z.device.type, enabled=False):
        w = affinities(z.float()[index])
        masks, values = [], []
        for i in index.tolist():
            s = Seeds.from_voxels(seed[i].cpu().numpy()[None], np.array([1]), tuple(label.shape[1:]), z.device)
            s = s.with_background(hu[i].float(), SEED_HU, SEED_EROSION)
            masks.append(s.mask)
            values.append(s.value)
        stats: list = []
        p = random_walker_batched(w, torch.stack(masks), torch.stack(values), options, stats)
        target = tree[index].float()
        p_safe = p.clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(p_safe, target)
        dice = soft_dice_loss(p, target)
        loss = bce + dice
    with torch.no_grad():
        hard = (p >= 0.5).float()
        dice_hard = (2 * (hard * target).flatten(1).sum(1) + 1) / (hard.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)
        other = (label[index] > 0).float() * (1 - target)
        leak = float((p * other).sum() / other.sum()) if float(other.sum()) > 0 else 0.0
        on_tree = float((p * target).sum() / target.sum().clamp_min(1))
        iterations = float(np.mean([s["iterations"] for s in stats if "iterations" in s])) if stats else math.nan
        numbers = dict(walker_bce=float(bce), walker_dice_loss=float(dice), dice=float(dice_hard.mean()), p_tree=on_tree, leak=leak,
                       iterations=iterations, converged=float(np.mean([s["converged"] for s in stats if "converged" in s])) if stats else math.nan)
    return loss, numbers


def total_loss(z: torch.Tensor, batch: dict, device: torch.device, walker_weight: float, options: SolverOptions) -> tuple[torch.Tensor, dict]:
    """Both losses on a batch: the affinity one, plus `walker_weight` times the walker's when
    that is positive."""
    label, hu = batch["label"].to(device), batch["hu"].to(device)
    loss, stats = affinity_loss(z, label, hu)
    if walker_weight > 0:
        walker, more = walker_loss(z, hu, label, batch["seed"].to(device), batch["tree"].to(device), options)
        if walker is not None:
            loss = loss + walker_weight * walker
            stats.update(more)
    return loss, stats


# ------------------------------------------------------------------ validation


def validation_loss(model: AffinityNet, loader, device: torch.device, walker_weight: float, options: SolverOptions, autocast) -> tuple[float, dict]:
    model.eval()
    losses, stats_all = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast():
                z = model(batch["image"].to(device, non_blocking=True))
            loss, stats = total_loss(z, batch, device, walker_weight, options)
            losses.append(float(loss))
            stats_all.append(stats)
    keys = set().union(*(s.keys() for s in stats_all)) if stats_all else set()
    return float(np.mean(losses)) if losses else math.nan, {k: _nanmean([s.get(k, math.nan) for s in stats_all]) for k in keys}


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else math.nan


def scan_dice(model: AffinityNet, extra: dict, cases: list, device: torch.device, n_clicks: int, seed: int) -> dict:
    """The simulated user on whole validation scans, segmenting as the editor would with the
    learned affinities (`evaluate.evaluate_scans`)."""
    model.eval()
    extractor = Learned(model, extra)  # runs the network where it is; the solver goes to the device the settings resolve
    settings = MRFSettings(feature_extractor="learned", device="cpu" if device.type == "cpu" else "auto")
    with torch.no_grad():
        return evaluate_scans(cases, settings, n_clicks, extractor, EXTRACTORS["learned"].affinities, seed, log=lambda s: None)


# ------------------------------------------------------------------ the run


def raise_open_file_limit() -> None:
    """Lift the soft limit on open files to the hard limit (often 1024 against 1 M): the workers
    open memory maps and shared-memory segments by the thousand over a run."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        elif hard == resource.RLIM_INFINITY and soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
    except (ImportError, ValueError, OSError):
        pass  # not Unix, or not allowed: the strategy above and the per-access maps are the real fix


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp(name: str, device: torch.device):
    """The autocast dtype: bf16 on Ampere and later, fp16 (with loss scaling, which the trainer
    adds) on older CUDA, None elsewhere."""
    if device.type != "cuda":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "off":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warm-up, then cosine to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


LOG_COLUMNS = ["step", "loss", "lumen_bce", "background_bce", "edge_accuracy", "dice", "p_tree", "leak", "iterations", "lr",
               "val_loss", "val_dice", "dice1", "dice3", "dice5", "leak1", "components1", "seconds"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="the 0.5 mm cache (absolute path, e.g. pseg3d's data/casx/cache)")
    ap.add_argument("--out", required=True, help="where the checkpoints and the log go")
    ap.add_argument("--steps", type=int, default=40000, help="optimiser steps in all, across the pieces of a run")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=1, help="batches per optimiser step")
    ap.add_argument("--patch", type=int, default=96, help="voxels a side")
    ap.add_argument("--window", type=int, default=96, help="voxels a side the editor runs the network in (saved with the checkpoint)")
    ap.add_argument("--walker-weight", type=float, default=0.0, help="weight of the walker's loss; 0 trains the affinities alone")
    ap.add_argument("--tolerance", type=float, default=1e-3, help="the walker's tolerance in training (the editor uses 1e-4)")
    ap.add_argument("--max-iterations", type=int, default=200, help="the walker's iteration cap in training")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--base", type=int, default=32, help="channels at full resolution")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--norm", default="channel", choices=["channel", "group"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-patches", type=int, default=16, help="fixed validation patches for the loss")
    ap.add_argument("--val-cases", type=int, default=4, help="whole validation scans the simulated user clicks on")
    ap.add_argument("--val-clicks", type=int, default=5)
    ap.add_argument("--train-cases", type=int, default=None, help="use only this many training scans (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="carry on from --out/last.pt: the network, the optimiser, the step, the best score")
    ap.add_argument("--init", default=None, help="start a new run from this checkpoint's weights (a fresh optimiser and schedule)")
    ap.add_argument("--checkpoint-every", type=int, default=None, help="steps between saves of last.pt (default: --val-every)")
    ap.add_argument("--time-limit", type=float, default=None, help="minutes: end this piece after that long, validating and saving")
    ap.add_argument("--stop-after", type=int, default=None, help="end this piece after this many steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Data-loader workers hand tensors to the main process through shared memory; the default
    # "file_descriptor" strategy leaks descriptors over a long run until "too many open files".
    torch.multiprocessing.set_sharing_strategy("file_system")
    raise_open_file_limit()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    amp = pick_amp(args.amp, device)
    torch.manual_seed(args.seed)
    options = SolverOptions(args.tolerance, args.max_iterations)

    cfg = PatchConfig(size=args.patch)
    train_set = PatchDataset(args.cache, "train", cfg, augment=True, seed=args.seed, limit=args.train_cases)
    val_set = PatchDataset(args.cache, "val", cfg, augment=False, seed=args.seed, limit=args.val_patches)
    val_cases = load_cases(args.cache, "val", args.val_cases)

    model = AffinityNet(UNet3DConfig(base=args.base, levels=args.levels, norm=args.norm)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    step, best = 0, -math.inf
    if args.resume:
        if not (out / "last.pt").is_file():
            raise SystemExit(f"--resume: no {out / 'last.pt'} to carry on from")
        blob = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        for name in ("patch", "base", "levels", "norm", "steps", "walker_weight"):
            if blob.get("args", {}).get(name) != getattr(args, name):
                print(f"warning: --{name.replace('_', '-')} is {getattr(args, name)}, but the run being resumed used {blob.get('args', {}).get(name)}", flush=True)
        model.load_state_dict(blob["state"])
        optimizer.load_state_dict(blob["optimizer"])
        if "scaler" in blob:
            scaler.load_state_dict(blob["scaler"])
        step, best = int(blob["step"]), float(blob.get("best", -math.inf))
        print(f"resumed {out / 'last.pt'} at step {step} (best {best:.3f})", flush=True)
    elif args.init:
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state"])
        print(f"initialised from {args.init} (step {blob.get('step', '?')} of its run)", flush=True)
    checkpoint_every = args.checkpoint_every or args.val_every

    def autocast():
        return torch.autocast(device.type, dtype=amp) if amp else torch.autocast(device.type, enabled=False)

    # Seeded by the step, so a resumed piece draws fresh patches rather than the last piece's.
    generator = torch.Generator()
    generator.manual_seed(args.seed * 1_000_003 + step)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.workers > 0, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=max(1, min(args.batch, 2)), num_workers=min(args.workers, 2))

    extra = dict(args=vars(args), spacing=float(train_set.cases[0].spacing), window=args.window)
    log_path = out / "log.csv"
    new_log = not (args.resume and log_path.is_file())
    log_file = open(log_path, "a" if not new_log else "w", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(LOG_COLUMNS)
    stage = "affinities" if args.walker_weight == 0 else f"affinities + {args.walker_weight:g} x walker"
    print(f"{parameter_count(model) / 1e6:.1f} M parameters, {len(train_set)} training scans, {len(val_cases)} validation scans, "
          f"batch {args.batch} x {args.accumulate}, {args.patch}^3 patches, loss: {stage}, on {device} "
          f"({ {torch.bfloat16: 'bf16', torch.float16: 'fp16 + loss scaling', None: 'fp32'}[amp] })", flush=True)

    def checkpoint(name: str, **more) -> None:
        """Written to a temporary file first and renamed into place, so a crash while saving
        leaves the previous checkpoint whole."""
        path = out / name
        save(model, str(path.with_suffix(".pt.tmp")), step=step, best=best, **extra, **more)
        os.replace(path.with_suffix(".pt.tmp"), path)

    def save_last() -> None:
        checkpoint("last.pt", optimizer=optimizer.state_dict(), scaler=scaler.state_dict())

    def validate() -> dict:
        v_loss, v_stats = validation_loss(model, val_loader, device, args.walker_weight, options, autocast)
        scans = scan_dice(model, extra, val_cases, device, args.val_clicks, args.seed)
        d = scans.get("dice", {})
        at = lambda k: d.get(str(k), math.nan)  # noqa: E731
        score = at(1) if not math.isnan(at(1)) else -v_loss
        after = ", ".join(f"{k}: {at(k):.3f}" for k in (1, 3, 5) if str(k) in d)
        print(f"  val @ {step}: loss {v_loss:.4f} edge accuracy {v_stats.get('edge_accuracy', math.nan):.3f} | simulated user on whole scans: "
              f"Dice after {after} clicks, leak {scans.get('leak1', math.nan):.3f}, pieces {scans.get('components1', math.nan):.1f}, "
              f"{scans.get('seconds_per_click', math.nan):.0f} s per click", flush=True)
        model.train()
        return dict(val_loss=v_loss, val_dice=v_stats.get("dice", math.nan), dice1=at(1), dice3=at(3), dice5=at(5),
                    leak1=scans.get("leak1", math.nan), components1=scans.get("components1", math.nan), score=score)

    started = time.perf_counter()
    running: dict[str, float] = {}
    counts: dict[str, int] = {}
    count = 0
    model.train()
    done = step >= args.steps
    stop_at = min(args.steps, step + args.stop_after) if args.stop_after else args.steps
    while not done:
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.steps, args.warmup, args.lr)
            with autocast():
                z = model(batch["image"].to(device, non_blocking=True))
            loss, stats = total_loss(z, batch, device, args.walker_weight, options)
            scaler.scale(loss / args.accumulate).backward()
            count += 1
            for k, v in dict(loss=float(loss.detach()), **stats).items():
                if not (isinstance(v, float) and math.isnan(v)):
                    running[k] = running.get(k, 0.0) + v
                    counts[k] = counts.get(k, 0) + 1
            if count % args.accumulate:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            out_of_time = args.time_limit is not None and (time.perf_counter() - started) / 60 >= args.time_limit
            last = step >= stop_at or out_of_time
            if step % checkpoint_every == 0 and not last:
                save_last()
            if step % args.log_every == 0 or last:
                row = {k: running[k] / counts[k] for k in running}
                running, counts, count = {}, {}, 0
                validated = validate() if (step % args.val_every == 0 or last) else None
                if validated and validated["score"] > best:
                    best = validated["score"]
                    checkpoint("best.pt", score=best)
                lr = lr_at(step, args.steps, args.warmup, args.lr)
                walker = f"  walker dice {row['dice']:.3f} p(tree) {row['p_tree']:.3f} leak {row['leak']:.3f} iters {row['iterations']:.0f}" if "dice" in row else ""
                print(f"step {step:6d}  loss {row.get('loss', math.nan):.4f}  edge bce {row.get('lumen_bce', math.nan):.4f} / {row.get('background_bce', math.nan):.4f}  "
                      f"edge acc {row.get('edge_accuracy', math.nan):.3f}{walker}  lr {lr:.2e}  {time.perf_counter() - started:.0f} s", flush=True)
                log.writerow([step, *(row.get(k, "") for k in LOG_COLUMNS[1:9]), lr,
                              *([validated[k] for k in ("val_loss", "val_dice", "dice1", "dice3", "dice5", "leak1", "components1")] if validated else [""] * 7),
                              time.perf_counter() - started])
                log_file.flush()
            if last:
                save_last()
                done = True
                break
    log_file.close()
    print("done" if step >= args.steps else f"paused at step {step} of {args.steps}: run again with --resume to carry on", flush=True)
    with open(out / "summary.json", "w") as f:
        json.dump(dict(step=step, steps=args.steps, best=best, finished=step >= args.steps), f, indent=1)


if __name__ == "__main__":
    main()
