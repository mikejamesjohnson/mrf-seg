# The model

What the editor computes, written out. Every symbol is defined where it first appears.

## 1. Setup

| Symbol | Meaning |
| --- | --- |
| $V$ | the voxels of an isotropic grid of spacing $h$ mm (0.5 mm by default: the image's finest voxel size, floored at 0.5, the grid the network is trained on) |
| $i, j$ | voxels, elements of $V$ |
| $p_i \in \mathbb R^3$ | the position of voxel $i$'s centre, LPS mm |
| $I$ | the CT image; $I_i$ is the HU value at voxel $i$ (resampled to the grid) |
| $y_i \in \{0, 1\}$ | the label of voxel $i$: 1 if its centre is in the lumen, 0 if not |
| $y = (y_i)_{i \in V}$ | a labeling of the whole grid |
| $\mathcal N(i)$ | the 6 face-neighbours of voxel $i$ |
| $\mathcal E$ | the edges: every unordered pair $\{i, j\}$ of neighbours, once |
| $\mathcal C = \{(k, \ell_k)\}$ | the points: a voxel $k$ and its label $\ell_k \in \{0, 1\}$ (inside = 1) |

The grid is the one the features are extracted on (`features.py`), the CRF is inferred on it
(`mrf.py`), and the maps are read back onto the CT's own grid trilinearly for display. The
6-connected grid has no triangles, so unary plus pairwise terms is the most general Markov model
on it; its two parities form a checkerboard (no two voxels of one parity are neighbours).

## 2. The model

The conditional random field is

$$
p(y \mid I) = \frac{1}{Z(I)} \exp\big(-U(y; I)\big), \qquad
U(y; I) = -\sum_{i \in V} a_i y_i + \sum_{\{i, j\} \in \mathcal E} w_{ij}\,[y_i \ne y_j],
$$

- $U$ is the energy of a labeling; lower is more probable;
- $Z(I) = \sum_{y'} \exp(-U(y'; I))$ is the partition function, a sum over all $2^{|V|}$ labelings;
- $a_i \in \mathbb R$ is the *unary logit* of voxel $i$: with no coupling, $p(y_i = 1) = \sigma(a_i)$, $\sigma(t) = 1 / (1 + e^{-t})$;
- $w_{ij} \ge 0$ is the *coupling* of edge $\{i, j\}$: the cost of giving the two neighbours different labels; $[\cdot]$ is 1 when the statement holds and 0 otherwise.

Any pairwise potential on binary labels reduces to this form (the rest moves into the unaries
or a constant), and $w_{ij} \ge 0$ makes $U$ submodular. The model is set by one number per voxel
and one per edge: a 4-channel map $(a, w_{+z}, w_{+y}, w_{+x})$ on the grid, with the last plane
along each axis unused. The image enters only through these numbers, so the potentials may use
any amount of context (a network's receptive field), and $p(I)$ is never modelled.

## 3. The points

A clicked voxel $k$ is *clamped*: $y_k = \ell_k$ throughout. The conditional of the other
voxels is the same kind of model, with each clamped neighbour $k$ of $i$ adding
$w_{ik}\,(2\ell_k - 1)$ to $a_i$. On its own a clamp reaches further than its neighbours only
through the coupling, one voxel per half-sweep (below). The points reach further in two other
ways: through the potentials, which are built from them (the Basic unary is a kernel density of
the points; the network takes click maps as input and is re-run around the points), and through
**connectivity**: after the sweeps, with the setting `connected` on, the face-connected
components of $\{m_i \ge 0.5\}$ that hold an inside point are kept and $m_i$ is set to 0 on the
others. An inside point thus selects the vessel it is in, as far as that vessel reaches through
voxels the model calls lumen, and an unclicked vessel drops out of the probability map, the
labelmap and the surface alike.

## 4. Inference: mean field

What is drawn is the marginal $p(y_i = 1 \mid I, \mathcal C)$, approximated by the closest product
distribution $q(y) = \prod_i q_i(y_i)$, $q_i(1) = m_i$, in $\mathrm{KL}(q \,\|\, p)$. Its stationary
point is

$$
m_i = \sigma\Big(a_i + \sum_{j \in \mathcal N(i)} w_{ij}\,(2 m_j - 1)\Big),
$$

the exact local conditional with the neighbours' labels replaced by their marginals. The two
parities are updated in turn: each half-sweep is an exact block coordinate minimisation of the
mean-field free energy

$$
F(q) = \mathbb E_q[U] - H(q)
= -\sum_i a_i m_i + \sum_{\{i,j\}} w_{ij}\,(m_i + m_j - 2 m_i m_j) + \sum_i \big(m_i \log m_i + (1 - m_i) \log (1 - m_i)\big)
\;\ge\; -\log Z,
$$

so $F$ never increases and the iteration converges. Clamped voxels are reset to their labels
after every half-sweep. The editor runs `sweeps` full sweeps from $m_i^{(0)} = \sigma(a_i)$ and
stops early once no marginal changes by more than $10^{-3}$ in a sweep (`mrf.mean_field`).

What mean field gets wrong: it is overconfident (the marginals are too sharp), and it is
conservative: a front advances into a region only where the unaries there are near zero or
positive, because undecided neighbours vote against. It cannot promise connectivity -- no pairwise
model can -- so the labelmap has a **Connected** option that keeps only the components an inside
point touches.

The labelmap is $[m_i \ge \tau]$ at a threshold $\tau$ (0.5), the surface is the $\tau$ level set
of $m$ (or the zero level set of $a$: the evidence alone).

## 5. Basic: handcrafted potentials

With the Basic extractor each voxel has a feature vector $\phi_i \in \mathbb R^C$: the image
blurred at $C$ scales ($\sigma = h, 2h, 4h, 8h$), each channel standardised over the image and
all scaled by $1/\sqrt C$ (optionally with the blurs' edge strength and Hessian eigenvalues).

**Background.** Lumen is a fraction of a percent of a CT, so a random sample of the image's
voxels is background. A Gaussian is fitted to their features: mean $\mu \in \mathbb R^C$ and
covariance $\Sigma$ (shrunk 10 % toward its diagonal). $\Sigma = L L^\top$ gives the whitening
$z_i = L^{-1}(\phi_i - \mu)$, so $|z_i|^2$ is the Mahalanobis distance of voxel $i$ from the
background.

**Unary.** A kernel density ratio in whitened feature space, the points' kernels multiplied by a
spatial kernel so that a point is local:

$$
\log p(\phi_i \mid \text{in}) = \log \frac{1}{|\mathcal C_{\text{in}}|} \sum_{k \in \mathcal C_{\text{in}}}
\exp\Big(-\frac{|z_i - z_k|^2}{2\kappa} - \frac{C}{2}\log\kappa - \frac{|p_i - p_k|^2}{2 \ell_s^2}\Big),
\qquad \kappa = \ell_f^2,
$$

$$
\log p(\phi_i \mid \text{bg}) = -\tfrac12 |z_i|^2, \qquad
p(\phi_i \mid \text{out}) = \begin{cases} p(\phi_i \mid \text{bg}) & \text{no outside points} \\ \tfrac12\big(p(\phi_i \mid \text{bg}) + p(\phi_i \mid \text{out points})\big) & \text{otherwise,}\end{cases}
$$

$$
a_i = a_0 + \beta\,\big[\log(p(\phi_i \mid \text{in}) + \varepsilon) - \log(p(\phi_i \mid \text{out}) + \varepsilon)\big],
$$

- $\ell_f$ (`feature_lengthscale`) is the kernel width in standard deviations of the image's features;
- $\ell_s$ (`spatial_lengthscale`, mm) is how far a point's evidence reaches;
- $a_0$ (`prior_logit`) shifts every logit; $\beta$ (`unary_scale`) scales the evidence against the coupling;
- $\varepsilon = 10^{-6}$ floors the densities, so a voxel like no point (far from every point in feature space or in position) gets a very negative logit: the unexplored image is background.

The Mahalanobis metric means a point reaches voxels that differ from it only in directions
along which the image varies anyway, and not in those where the point itself is unusual -- the
coarse-scale channels, where a thin vessel is dim and a chamber is not. On 8 test cubes of
96³ voxels with a simulated user, the Dice after 1 / 3 / 5 clicks was 0.42 / 0.53 / 0.58 with no
leak into other vessels: precise, short of recall.

**Coupling.** Contrast-sensitive, in HU (Boykov & Jolly):

$$
w_{ij} = \lambda \Big(f + (1 - f)\, e^{-(I_i - I_j)^2 / 2\delta^2}\Big),
$$

with $\lambda$ (`coupling`), $\delta$ (`contrast`, HU) and $f \in [0, 1]$ (`coupling_floor`). An
edge with a step of $\delta$ keeps $e^{-1/2} = 61\,\%$ of the coupling; of $3\delta$, 1 %. The grid
is isotropic, so no distance factor is needed. The same cost applies in every direction, which is
the shrinking bias of §6.

## 6. Learned: the network's potentials

A 3D U-Net $g_\theta$ (five resolutions, 22.6 M parameters, normalised per voxel over its
channels so windows of a volume agree where they overlap) takes three channels per voxel -- the
HU volume on the fixed window $(\mathrm{clip}(\mathrm{HU}, -200, 1000) - 100) / 300$, and two
*click maps* $c^{\text{in}}, c^{\text{out}}$, each the largest Gaussian mark
$\exp(-|p_i - p_k|^2 / 2\sigma^2)$ ($\sigma$ = 2 voxels) of any point $k$ of that label -- and
gives four channels per voxel:

$$
a_i = g_\theta(I, c)_{i,0} + s, \qquad w_{i,+d} = \lambda \cdot \mathrm{softplus}\big(g_\theta(I, c)_{i,d}\big), \quad d \in \{z, y, x\},
$$

- $s$ (`unary_shift`) and $\lambda$ (`coupling`) are the editor's knobs, 0 and 1 as trained;
- softplus keeps every coupling non-negative, so the energy stays submodular;
- the head's unary bias starts at $-4$ (lumen is rare) and its coupling bias where softplus gives 0.1: small, so the first sweeps are nearly the identity and the gradient reaches the unary; with couplings near 1 the mean field saturates at the all-background consensus and the network learns to smooth harder instead of finding lumen.

The click maps are hints, not potentials: what a point selects, and how far along what, is the
network's to decide from the image, so a point in one of two vessels 2 mm apart need not touch
the other. Its reach is bounded by the receptive field (about 50 mm); beyond that, §3's
connectivity carries the selection along the vessel.

At session open the network runs once over the whole image with empty click maps, in overlapping
96³ windows blended with Gaussian weights: the automatic potentials, kept as the image's
"features". A Generate with points re-runs it only in 96³ windows centred on the points (one
window per cluster of points within a quarter window), with the click maps as input, and each
window's output replaces the automatic one at its centre and fades into it toward its faces.
The couplings can differ by direction, which is what lets the model be cheap to cut across a
wall and expensive to cut along a vessel -- the fix for the shrinking bias: a tube of radius $r$
voxels survives isotropic smoothing of strength $w$ against a unary $a$ only if $r > 2w/a$, and
coronary lumen is about 2 voxels in radius at 0.5 mm.

## 7. Training

Patches of 96³ voxels are cut from the 0.5 mm cache (560 training scans), mostly centred near
vessels, with the 48 symmetries of the cube and a little HU jitter as augmentation. For each
patch a user's clicks are simulated: 0-3 inside (spread over the lumen's connected components),
0-4 outside (three in ten on a vessel that got no inside click, the rest mostly within 2 mm of
the wall), and a fifth of the patches get none. The target $y$ is the lumen *as the clicks
condition it*: with inside clicks, the 26-connected components that hold one; without, all of
it; less any component that holds an outside click and no inside one. So the network is trained
to read a click as "select the structure I am in" or "drop the structure I am in", and to
segment every coronary when there is none. With $m = \mathrm{MF}_T(a, w, \mathcal C)$ the
marginals after $T = 20$ sweeps with the clicks clamped, differentiable in $a$ and $w$
(`mrf.mean_field_batched`), the loss is

$$
\mathcal L = \mathrm{BCE}(m, y) + \mathrm{Dice}_{\text{soft}}(m, y) + \tfrac12\,\mathrm{BCE}(\sigma(a), y),
$$

- $\mathrm{BCE}$ is the binary cross-entropy over the voxels;
- $\mathrm{Dice}_{\text{soft}} = 1 - \frac{2\sum_i m_i y_i + 1}{\sum_i m_i + \sum_i y_i + 1}$ keeps the rare structure from being ignored;
- the last term keeps the unary a usable classifier on its own (the editor shows it, and the inference starts from it).

The gradient reaches $\theta$ through the sweeps. AdamW, learning rate $3 \cdot 10^{-4}$ with a
1000-step warm-up and a cosine decay to a tenth, weight decay $10^{-5}$, gradient clipping at 1,
batch 4 in bf16 on a 24 GB GPU, 40 000 steps. Validation every 2000 steps: the loss on fixed
patches, and a simulated user on sub-volumes of validation scans segmented exactly as the editor
would (Dice with 0, 1, 3 and 5 clicks, and the leak into other vessels); `best.pt` is the
checkpoint with the best mean of those Dice values.

## 8. What the model does not do, yet

Connectivity inside the inference. Two coronary trees meet only through the aorta, which is not
labelled, so the label maps have two 6-connected components (sometimes three or four, from
gaps), no loops and no cavities. A pairwise CRF cannot express "connected to a root"; the
`connected` rule of §3 enforces it after the sweeps, on the marginals. The principled version is
a tree constraint -- a shortest-path forest from the inside points with the submodular potential
$\infty \cdot [y_i = 1, y_{\pi(i)} = 0]$ on each voxel and its parent $\pi(i)$ -- which keeps the
MAP labeling exactly solvable by min-cut and, on the tree alone, gives exact marginals in two
passes; it would also let a click bridge a gap the model leaves at a stenosis, which the rule of
§3 cannot. That is the next step after the network.
