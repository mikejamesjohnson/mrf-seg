"""The network: a 3D U-Net from the image (`learned.network_input`) to three output channels at
every voxel, the log-affinities of its edges to the neighbour along +z, +y and +x, which
`rw.affinities` exponentiates (and clamps) into the walker's graph.

Five resolutions (a 96^3 patch: 96 -> 48 -> 24 -> 12 -> 6), two 3x3x3 convolutions per level
with a normalisation and a GELU after each, max-pooling down, transposed convolutions up, skip
connections by concatenation, and a 1x1x1 head to 3 channels. Every layer is a local rule
applied at every position, so the network is defined on a patch and runs on a volume of any
size (padded to a multiple of 16, the coarsest stride); the normalisation is over the channels
of each voxel (`ChannelNorm`) and not over the patch, so the affinities of a voxel depend on its
surroundings alone and windows of a volume agree where they overlap. GroupNorm, whose
statistics are the window's, is kept as `norm="group"` for an ablation.

The head's bias starts at 0: every edge begins at affinity 1, a uniform graph on which the
walker's probability is a smooth blob about the seed, and the first gradients say where to cut
and where to join.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..rw import AXES
from . import IN_CHANNELS, SPACING

#: Output channels: one log-affinity per axis.
OUT_CHANNELS = AXES


@dataclass(frozen=True)
class UNet3DConfig:
    inputs: int = IN_CHANNELS  # input channels: the image
    base: int = 32          # channels at full resolution, doubled at each level down
    levels: int = 5         # resolutions: 96 -> 48 -> 24 -> 12 -> 6 on a 96^3 patch
    norm: str = "channel"   # "channel": per voxel over its channels; "group": GroupNorm(groups)
    groups: int = 8         # GroupNorm groups, when norm is "group"

    @property
    def stride(self) -> int:
        """The coarsest stride: input sides are padded to a multiple of it."""
        return 2 ** (self.levels - 1)


class _ChannelNormFunction(torch.autograd.Function):
    """The normalisation with its own backward, so that only the input (in its own dtype -- fp16
    under autocast) and the per-voxel mean and 1/std are kept for it. Left to autograd, the
    centred activations would be kept in float32 as well: on a 96^3 patch at batch 2 that is
    about a gigabyte more per norm layer at full resolution."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        x32 = x.to(torch.promote_types(x.dtype, torch.float32))     # half precision up to float32; float64 stays
        mean = x32.mean(1, keepdim=True)
        rstd = torch.rsqrt(x32.var(1, unbiased=False, keepdim=True) + eps)
        y = (x32 - mean) * rstd * weight.view(shape) + bias.view(shape)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        shape = (1, -1) + (1,) * (x.dim() - 2)
        reduce = [0] + list(range(2, x.dim()))
        wide = torch.promote_types(x.dtype, torch.float32)
        dy32 = dy.to(wide)
        xhat = (x.to(wide) - mean) * rstd
        dweight = (dy32 * xhat).sum(reduce)
        dbias = dy32.sum(reduce)
        g = dy32 * weight.view(shape)
        dx = rstd * (g - g.mean(1, keepdim=True) - xhat * (g * xhat).mean(1, keepdim=True))
        return dx.to(x.dtype), dweight, dbias, None


class ChannelNorm(nn.Module):
    """LayerNorm over the channel axis at each voxel: x (B, C, ...) -> (x - mean_c) / std_c, then an
    affine per channel. Reads nothing but the voxel itself (ConvNeXt's channels-first LayerNorm).
    The statistics are taken in float32 whatever the input's precision."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ChannelNormFunction.apply(x, self.weight, self.bias, self.eps)

    @staticmethod
    def reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """The same, written out for autograd: what the tests check the Function against."""
        shape = (1, -1) + (1,) * (x.dim() - 2)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight.view(shape) + bias.view(shape)


def _norm(cfg: UNet3DConfig, channels: int) -> nn.Module:
    if cfg.norm == "channel":
        return ChannelNorm(channels)
    if cfg.norm == "group":
        return nn.GroupNorm(cfg.groups, channels)
    raise ValueError(f"norm must be 'channel' or 'group', not {cfg.norm!r}")


class _Block(nn.Module):
    """Two 3x3x3 convolutions, each normalised and activated. Zero padding: the input is scaled so
    that zero is soft tissue."""

    def __init__(self, cin: int, cout: int, cfg: UNet3DConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _norm(cfg, cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _norm(cfg, cout), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, cfg: UNet3DConfig) -> None:
        super().__init__()
        chans = [cfg.base * 2 ** k for k in range(cfg.levels)]
        self.down = nn.ModuleList([_Block(cfg.inputs, chans[0], cfg)] + [_Block(chans[k - 1], chans[k], cfg) for k in range(1, cfg.levels)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(chans[k], chans[k - 1], 2, stride=2) for k in range(cfg.levels - 1, 0, -1)])
        self.merge = nn.ModuleList([_Block(2 * chans[k - 1], chans[k - 1], cfg) for k in range(cfg.levels - 1, 0, -1)])
        self.head = nn.Conv3d(chans[0], OUT_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, block in enumerate(self.down):
            x = block(F.max_pool3d(x, 2) if k else x)
            skips.append(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x).float()


class AffinityNet(nn.Module):
    """The U-Net on any volume size: each side is zero-padded to a multiple of the network's
    coarsest stride, split between both ends, and the output is cropped back. The output is the
    log-affinities (B, 3, z, y, x), float32: `rw.affinities` turns one volume of it into the
    walker's graph."""

    def __init__(self, cfg: UNet3DConfig = UNet3DConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = UNet3D(cfg)
        with torch.no_grad():
            self.net.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, inputs, z, y, x), the normalised HU (`learned.network_input`) -> (B, 3, z, y, x)
        log-affinities, float32."""
        stride = self.cfg.stride
        sizes = x.shape[-3:]
        pads = [(-n) % stride for n in sizes]
        before = [p // 2 for p in pads]
        if any(pads):
            x = F.pad(x, (before[2], pads[2] - before[2], before[1], pads[1] - before[1], before[0], pads[0] - before[0]))
        z = self.net(x)
        return z[..., before[0]:before[0] + sizes[0], before[1]:before[1] + sizes[1], before[2]:before[2] + sizes[2]]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save(model: AffinityNet, path: str, **extra) -> None:
    """The model with what the extractor needs to run it: its config, the grid spacing (`SPACING`
    unless `extra` says otherwise) and whatever else is given (the training's arguments, the step)."""
    extra.setdefault("spacing", SPACING)
    torch.save(dict(config=asdict(model.cfg), state=model.state_dict(), **extra), path)


def load(path: str, device: str | torch.device = "cpu") -> tuple[AffinityNet, dict]:
    """The model saved at `path`, in eval mode on `device`, and whatever else was saved with it
    (including `spacing`)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(blob["config"])
    config.setdefault("inputs", int(blob["state"]["net.down.0.net.0.weight"].shape[1]))
    model = AffinityNet(UNet3DConfig(**config))
    if int(blob["state"]["net.head.weight"].shape[0]) != OUT_CHANNELS:
        raise ValueError(f"{path} is not an affinity network: its head has {int(blob['state']['net.head.weight'].shape[0])} outputs, not {OUT_CHANNELS}")
    model.load_state_dict(blob["state"])
    return model.to(device).eval(), {k: v for k, v in blob.items() if k not in ("config", "state")}
