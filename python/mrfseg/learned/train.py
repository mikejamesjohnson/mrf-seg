"""Train the potentials network on the cache `prepare.py` wrote (or pseg3d's, by absolute path).

    python -m mrfseg.learned.train --cache /path/to/pseg3d/data/casx/cache --out runs/first [--steps 40000 --batch 4 ...]

Every `--log-every` steps it prints the loss and the Dice of the mean-field marginals on the
training patches; every `--val-every` steps the loss on fixed validation patches, and a
simulated user's Dice after 0 / 1 / 3 / 5 clicks on `--val-cube`-voxel sub-volumes of
`--val-cases` validation scans, segmented exactly as the editor would segment them. `best.pt` is
the checkpoint with the best mean of those Dice values, `last.pt` the latest, with the optimiser.
A run can be done in pieces: `--time-limit MINUTES` or `--stop-after STEPS` ends one, validating
and saving where it got to, and `--resume` carries on from there with fresh patches (the data
pipeline is seeded by the step, so a piece never replays the last one's draws) on the schedule
`--steps` defines -- keep that the same across the pieces. On CUDA the network runs in bf16
(Ampere and later) or fp16 with loss scaling (Turing and older), the inference and the loss in
fp32; `--accumulate` trades batch size for memory, and `--checkpoint-sweeps` recomputes each
mean-field sweep in the backward pass instead of keeping it. The defaults -- batch 4, 96^3
patches, bf16 -- are for a 24 GB GPU; an 8 GB one takes `--batch 1 --accumulate 4`.

The loss (`segmentation_loss`): the marginals after `--sweeps` mean-field sweeps through the CRF
the network's output defines, with the item's simulated clicks clamped, are scored against the
lumen label with binary cross-entropy plus a soft Dice (lumen is a fraction of a percent of the
voxels, and the Dice keeps the small structure from being ignored); and the unary's own sigmoid
is scored with a lighter cross-entropy, so it is a usable classifier by itself (the editor shows
it, and the inference starts from it). The gradient reaches the network through the sweeps.
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
from ..mrf import mean_field_batched
from ..settings import MRFSettings
from .clicks import simulate
from .data import MAX_CLICKS, PatchConfig, PatchDataset, load_cases
from .evaluate import segmenter_for, targets_in
from .extractor import Learned
from .model import PotentialNet, UNet3DConfig, parameter_count, save

# ------------------------------------------------------------------ the loss


def clamps_of(clicks: torch.Tensor, shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense clamps from (B, MAX_CLICKS, 4) rows of (z, y, x, label), label 0 being padding:
    the mask (B, z, y, x) and the values there."""
    b = clicks.shape[0]
    mask = torch.zeros((b, *shape), dtype=torch.bool, device=clicks.device)
    value = torch.zeros((b, *shape), dtype=torch.float32, device=clicks.device)
    for i in range(b):
        rows = clicks[i][clicks[i, :, 3] != 0]
        if len(rows):
            mask[i, rows[:, 0], rows[:, 1], rows[:, 2]] = True
            value[i, rows[:, 0], rows[:, 1], rows[:, 2]] = (rows[:, 3] > 0).float()
    return mask, value


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - the soft Dice of each item, averaged: (B, ...) probabilities against (B, ...) 0/1."""
    p = probability.flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def segmentation_loss(raw: torch.Tensor, label: torch.Tensor, clicks: torch.Tensor, sweeps: int, unary_weight: float = 0.5,
                      checkpoint: bool = False) -> tuple[torch.Tensor, dict]:
    """The loss of a batch: `raw` (B, 4, z, y, x) from the network, `label` (B, z, y, x) 0/1,
    `clicks` (B, MAX_CLICKS, 4). Returns it and a few numbers worth logging."""
    raw = raw.float()
    target = label.float()
    unary = raw[:, 0]
    coupling = F.softplus(raw[:, 1:])
    mask, value = clamps_of(clicks, tuple(unary.shape[1:]))
    marginal = mean_field_batched(unary, coupling, mask, value, sweeps, checkpoint)
    marginal = marginal.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(marginal, target)
    dice = soft_dice_loss(marginal, target)
    unary_bce = F.binary_cross_entropy_with_logits(unary, target)
    loss = bce + dice + unary_weight * unary_bce
    with torch.no_grad():
        hard = (marginal > 0.5).float()
        dice_hard = (2 * (hard * target).flatten(1).sum(1) + 1) / (hard.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)
        stats = dict(bce=float(bce), dice_loss=float(dice), unary_bce=float(unary_bce), dice=float(dice_hard.mean()),
                     coupling=float(coupling.mean()), lumen=float(target.mean()))
    return loss, stats


# ------------------------------------------------------------------ validation


def validation_loss(model: PotentialNet, loader, device: torch.device, sweeps: int, autocast) -> tuple[float, dict]:
    model.eval()
    losses, stats_all = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast():
                raw = model(batch["image"].to(device, non_blocking=True))
            loss, stats = segmentation_loss(raw, batch["label"].to(device), batch["clicks"].to(device), sweeps)
            losses.append(float(loss))
            stats_all.append(stats)
    keys = stats_all[0].keys() if stats_all else []
    return float(np.mean(losses)) if losses else math.nan, {k: float(np.mean([s[k] for s in stats_all])) for k in keys}


def click_dice(model: PotentialNet, extra: dict, cubes: list[dict], device: torch.device, n_clicks: int = 5) -> dict:
    """Mean Dice with 0 clicks and after each of `n_clicks` simulated clicks, segmenting each
    cube as the editor would (`evaluate.segmenter_for`, with the learned potentials), and the
    leak into other lumen after the last."""
    model.eval()
    extractor = Learned(model, extra)  # runs the network where it is; the inference goes to the device the settings resolve
    settings = MRFSettings(device="cpu" if device.type == "cpu" else "auto", **extractor.setting_defaults)
    zero, dices, leaks = [], [], []
    with torch.no_grad():
        for cube in cubes:
            segment = segmenter_for(cube, settings, extractor, EXTRACTORS["learned"].potentials)
            prediction = segment(np.zeros((0, 4), dtype=np.int64))
            zero.append(_dice(prediction, cube["target"]))
            result = simulate(segment, cube["target"], n_clicks, others=cube["others"], first=cube["at"])
            dices.append(result["dice"])
            leaks.append(result["leak"][-1] if result["leak"] else 0.0)
    return dict(dice0=float(np.mean(zero)) if zero else math.nan, dice=np.mean(dices, axis=0).tolist() if dices else [],
                leak=float(np.mean(leaks)) if leaks else 0.0)


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    total = int(prediction.sum()) + int(target.sum())
    return 1.0 if total == 0 else 2.0 * int((prediction & target).sum()) / total


# ------------------------------------------------------------------ the run


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="the 0.5 mm cache (absolute path, e.g. pseg3d's data/casx/cache)")
    ap.add_argument("--out", required=True, help="where the checkpoints and the log go")
    ap.add_argument("--steps", type=int, default=40000, help="optimiser steps in all, across the pieces of a run")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=1, help="batches per optimiser step")
    ap.add_argument("--patch", type=int, default=96, help="voxels a side")
    ap.add_argument("--sweeps", type=int, default=20, help="mean-field sweeps the loss is taken through (and the editor starts from)")
    ap.add_argument("--checkpoint-sweeps", action="store_true", help="recompute each sweep in the backward pass, to save memory")
    ap.add_argument("--unary-weight", type=float, default=0.5, help="weight of the unary's own cross-entropy")
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
    ap.add_argument("--val-cases", type=int, default=8, help="validation scans the simulated user clicks on")
    ap.add_argument("--val-cube", type=int, default=96)
    ap.add_argument("--val-clicks", type=int, default=5)
    ap.add_argument("--train-cases", type=int, default=None, help="use only this many training scans (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="carry on from --out/last.pt: the network, the optimiser, the step, the best score")
    ap.add_argument("--init", default=None, help="start a new run from this checkpoint's weights (a fresh optimiser and schedule)")
    ap.add_argument("--checkpoint-every", type=int, default=None, help="steps between saves of last.pt (default: --val-every)")
    ap.add_argument("--time-limit", type=float, default=None, help="minutes: end this piece after that long, validating and saving")
    ap.add_argument("--stop-after", type=int, default=None, help="end this piece after this many steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    amp = pick_amp(args.amp, device)
    torch.manual_seed(args.seed)

    cfg = PatchConfig(size=args.patch)
    train_set = PatchDataset(args.cache, "train", cfg, augment=True, seed=args.seed, limit=args.train_cases)
    val_set = PatchDataset(args.cache, "val", cfg, augment=False, seed=args.seed, limit=args.val_patches)
    val_cubes = targets_in(load_cases(args.cache, "val", args.val_cases), 1, args.val_cube, args.seed)

    model = PotentialNet(UNet3DConfig(base=args.base, levels=args.levels, norm=args.norm)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    step, best = 0, -math.inf
    if args.resume:
        if not (out / "last.pt").is_file():
            raise SystemExit(f"--resume: no {out / 'last.pt'} to carry on from")
        blob = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        for name in ("patch", "sweeps", "base", "levels", "norm", "steps"):
            if blob.get("args", {}).get(name) != getattr(args, name):
                print(f"warning: --{name} is {getattr(args, name)}, but the run being resumed used {blob.get('args', {}).get(name)}", flush=True)
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

    extra = dict(args=vars(args), spacing=float(train_set.cases[0].spacing))
    log_path = out / "log.csv"
    new_log = not (args.resume and log_path.is_file())
    log_file = open(log_path, "a" if not new_log else "w", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "loss", "dice", "bce", "dice_loss", "unary_bce", "coupling", "lr", "val_loss", "val_dice", "dice0", "dice1", "dice3", "dice5", "leak5", "seconds"])
    print(f"{parameter_count(model) / 1e6:.1f} M parameters, {len(train_set)} training scans, {len(val_cubes)} validation cubes, "
          f"batch {args.batch} x {args.accumulate}, {args.sweeps} sweeps, on {device} "
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
        v_loss, v_stats = validation_loss(model, val_loader, device, args.sweeps, autocast)
        clicks = click_dice(model, extra, val_cubes, device, args.val_clicks)
        d = clicks["dice"]
        at = lambda k: d[k - 1] if len(d) >= k else math.nan  # noqa: E731
        score = float(np.nanmean([clicks["dice0"], at(1), at(3), at(5)])) if d else -v_loss
        print(f"  val @ {step}: loss {v_loss:.4f} dice {v_stats.get('dice', math.nan):.3f} | simulated user: "
              f"0 clicks {clicks['dice0']:.3f}, 1: {at(1):.3f}, 3: {at(3):.3f}, 5: {at(5):.3f}, leak {clicks['leak']:.3f}", flush=True)
        model.train()
        return dict(val_loss=v_loss, val_dice=v_stats.get("dice", math.nan), dice0=clicks["dice0"], dice1=at(1), dice3=at(3), dice5=at(5), leak5=clicks["leak"], score=score)

    started = time.perf_counter()
    running: dict[str, float] = {}
    count = 0
    model.train()
    done = step >= args.steps
    stop_at = min(args.steps, step + args.stop_after) if args.stop_after else args.steps
    while not done:
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.steps, args.warmup, args.lr)
            with autocast():
                raw = model(batch["image"].to(device, non_blocking=True))
            loss, stats = segmentation_loss(raw, batch["label"].to(device), batch["clicks"].to(device), args.sweeps, args.unary_weight, args.checkpoint_sweeps)
            scaler.scale(loss / args.accumulate).backward()
            count += 1
            for k, v in dict(loss=float(loss.detach()), **stats).items():
                running[k] = running.get(k, 0.0) + v
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
                n = max(count, 1)
                row = {k: v / n for k, v in running.items()}
                running, count = {}, 0
                validated = validate() if (step % args.val_every == 0 or last) else None
                if validated and validated["score"] > best:
                    best = validated["score"]
                    checkpoint("best.pt", score=best)
                print(f"step {step:6d}  loss {row['loss']:.4f}  dice {row['dice']:.3f}  bce {row['bce']:.4f}  unary bce {row['unary_bce']:.4f}  "
                      f"coupling {row['coupling']:.2f}  lr {lr_at(step, args.steps, args.warmup, args.lr):.2e}  {time.perf_counter() - started:.0f} s", flush=True)
                log.writerow([step, row["loss"], row["dice"], row["bce"], row["dice_loss"], row["unary_bce"], row["coupling"], lr_at(step, args.steps, args.warmup, args.lr),
                              *([validated[k] for k in ("val_loss", "val_dice", "dice0", "dice1", "dice3", "dice5", "leak5")] if validated else [""] * 7),
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
