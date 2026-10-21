"""A simulated user who clicks toward a target segmentation -- for validation while training,
for the evaluation, and for the training itself (the clicks a patch is trained with are drawn
the same way a user would place them, more or less).

The protocol is the standard one of interactive segmentation: each click goes at the centre
(the voxel farthest from the edge) of the largest region where the segmentation is wrong --
inside if that region is missed lumen, outside if it is a false one -- so the first click lands
in the middle of the target. What segments is a callable from clicks to a prediction, so the
same user tests the handcrafted potentials and the learned ones alike.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

#: Clicks as rows of (z, y, x, label), label +1 inside and -1 outside; a prediction as a bool (z, y, x).
Segmenter = Callable[[np.ndarray], np.ndarray]


def next_click(prediction: np.ndarray, target: np.ndarray, within: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    """(z, y, x, label) for the next click, or None once nothing is wrong. With `within`, only
    errors there count (the rest of the volume is nobody's business)."""
    wrong = prediction != target
    if within is not None:
        wrong &= within
    if not wrong.any():
        return None
    labels, n = ndimage.label(wrong)
    sizes = ndimage.sum(wrong, labels, index=np.arange(1, n + 1))
    region = labels == (int(np.argmax(sizes)) + 1)
    depth = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1, 1:-1]
    z, y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return int(z), int(y), int(x), 1 if target[z, y, x] else -1


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    total = int(prediction.sum()) + int(target.sum())
    return 1.0 if total == 0 else 2.0 * int((prediction & target).sum()) / total


def simulate(segment: Segmenter, target: np.ndarray, n_clicks: int = 10, others: np.ndarray | None = None,
             within: np.ndarray | None = None, first: tuple[int, int, int] | None = None) -> dict:
    """Click `n_clicks` times toward `target`, segmenting after each with `segment`. Dice after
    each click (over `within`, if given), the clicks, and -- with `others`, the lumen of every
    other structure -- how much of it the segmentation took in after each click (`leak`). The
    first click goes at `first` if given (inside), else where the protocol puts it."""
    prediction = np.zeros(target.shape, dtype=bool)
    scope = within if within is not None else np.ones(target.shape, dtype=bool)
    clicks: list[tuple[int, int, int, int]] = []
    dices, leaks = [], []
    for n in range(n_clicks):
        click = (*first, 1) if (n == 0 and first is not None) else next_click(prediction, target, within)
        if click is not None:
            clicks.append(click)
            prediction = np.asarray(segment(np.array(clicks)), dtype=bool)
        dices.append(dice(prediction & scope, target & scope))
        if others is not None:
            leaks.append(float((prediction & others).sum() / max(int(others.sum()), 1)))
    return dict(dice=dices, clicks=clicks, leak=leaks)


def clicks_to_reach(dices: list[float], level: float) -> int:
    """How many clicks it took to reach Dice `level`: len(dices) + 1 if it never did."""
    reached = np.flatnonzero(np.asarray(dices) >= level)
    return int(reached[0]) + 1 if reached.size else len(dices) + 1


#: The connectivity a click's structure is taken in: 26 neighbours, so a vessel one voxel thick
#: that steps diagonally is still one structure.
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def random_clicks(lumen: np.ndarray, rng: np.random.Generator, n_inside: int, n_outside: int, spacing: float,
                  near_wall: float = 2.0, near_share: float = 2 / 3, margin: int = 0, on_lumen_share: float = 0.3) -> np.ndarray:
    """Clicks for training, (n, 4) rows of (z, y, x, label): `n_inside` in the `lumen` (binary
    (z, y, x)), spread over its connected components in turn so a small branch gets its share,
    and `n_outside` outside it -- `on_lumen_share` of those on a lumen component that got no
    inside click, where a user clicks to drop a vessel that is not wanted, and of the rest
    `near_share` within `near_wall` mm of the lumen, where a user's outside clicks go. All keep
    `margin` voxels from the volume's faces. Fewer when there is no room (no lumen: no inside
    clicks)."""
    s = np.array(lumen.shape)
    inner = np.zeros_like(lumen)
    inner[tuple(slice(margin, n - margin) for n in s)] = True
    rows: list[tuple[int, int, int, int]] = []
    components, count = ndimage.label(lumen, structure=STRUCTURE) if lumen.any() else (np.zeros_like(lumen, dtype=np.int32), 0)
    clicked: set[int] = set()
    if n_inside and (lumen & inner).any():
        pools = [(c, np.argwhere((components == c) & inner)) for c in range(1, count + 1)]
        pools = sorted([p for p in pools if len(p[1])], key=lambda p: len(p[1]), reverse=True)
        for k in range(n_inside):
            c, pool = pools[k % len(pools)]
            z, y, x = pool[rng.integers(len(pool))]
            rows.append((int(z), int(y), int(x), 1))
            clicked.add(c)
    outside = ~lumen & inner
    if n_outside and outside.any():
        near = outside & (ndimage.distance_transform_edt(~lumen, sampling=spacing) <= near_wall) if lumen.any() else np.zeros_like(lumen)
        near_voxels, any_voxels = np.argwhere(near), np.argwhere(outside)
        unwanted = np.argwhere(lumen & inner & ~np.isin(components, list(clicked))) if count > len(clicked) else np.zeros((0, 3), dtype=np.int64)
        for _ in range(n_outside):
            if len(unwanted) and rng.random() < on_lumen_share:
                pool = unwanted
            else:
                pool = near_voxels if len(near_voxels) and rng.random() < near_share else any_voxels
            z, y, x = pool[rng.integers(len(pool))]
            rows.append((int(z), int(y), int(x), -1))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def conditioned_target(lumen: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """What the segmentation should be given the clicks: with inside clicks, the connected
    components of `lumen` (binary (z, y, x)) that hold one; without, all of it; less any
    component that holds an outside click and no inside click. (n, 4) `clicks` as `random_clicks`
    gives them, label 0 being padding."""
    clicks = np.asarray(clicks).reshape(-1, 4)
    clicks = clicks[clicks[:, 3] != 0]
    if len(clicks) == 0 or not lumen.any():
        return lumen.copy()
    components, count = ndimage.label(lumen, structure=STRUCTURE)
    at = components[clicks[:, 0], clicks[:, 1], clicks[:, 2]]
    inside = set(int(c) for c in at[clicks[:, 3] > 0] if c > 0)
    outside = set(int(c) for c in at[clicks[:, 3] < 0] if c > 0) - inside
    if inside:
        keep = inside
    else:
        keep = set(range(1, count + 1)) - outside
    return np.isin(components, list(keep)) if keep else np.zeros_like(lumen)
