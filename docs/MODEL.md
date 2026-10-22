# The model

What the editor computes, written out. Every symbol is defined where it first appears. The
formulation is the random walker (Grady 2006) on *learned* affinities: a Gaussian Markov random
field over the voxels, conditioned exactly on the user's points. Section 10 says what it
replaced and why.

## 1. Setup

| Symbol | Meaning |
| --- | --- |
| $V$ | the voxels of an isotropic grid of spacing $h$ mm (0.5 mm by default: the image's finest voxel size, floored at 0.5, the grid the network is trained on) |
| $i, j$ | voxels, elements of $V$; $N = \lvert V \rvert$ |
| $I$ | the CT image; $I_i$ is the Hounsfield value at voxel $i$ (resampled to the grid) |
| $y_i \in \{0, 1\}$ | the label of voxel $i$ in the training data: 1 if its centre is in the lumen, 0 if not |
| $\mathcal N(i)$ | the 6 face-neighbours of voxel $i$ |
| $\mathcal E$ | the edges: every unordered pair $\{i, j\}$ of neighbours, once; $i \sim j$ means $\{i, j\} \in \mathcal E$ |
| $w_{ij} > 0$ | the *affinity* of edge $\{i, j\}$: how freely the label flows between the two voxels; $w_{ij} = w_{ji}$ |
| $z_{ij} = \log w_{ij}$ | its *log-affinity*, what is stored and learned |
| $d_i = \sum_{j \sim i} w_{ij}$ | the *degree* of voxel $i$ |
| $S \subset V$ | the *seeds*: voxels whose value is fixed; $F = V \setminus S$ the free voxels |
| $s_k \in \{0, 1\}$ | the value of seed $k$: 1 at an inside point, 0 at an outside point or a background seed |
| $p_i \in [0, 1]$ | the probability the editor draws at voxel $i$ |

The grid is the one the features are extracted on (`features.py`), the walker is solved on it
(`rw.py`), and the maps are read back onto the CT's own grid trilinearly for display. The
label maps of the training data are face-connected: one component per coronary tree (the two
trees meet only through the aorta, which is not labelled), with 99 % or more of the lumen in the
largest 6-connected component of each tree, so the 6-connected graph is the right one for
"the vessel a click is in".

## 2. The model

The field is real-valued, $u \in \mathbb R^N$, with the Gaussian Markov random field

$$
p(u) \propto \exp\!\big(-\tfrac12 E(u)\big), \qquad
E(u) = \sum_{\{i,j\} \in \mathcal E} w_{ij}\,(u_i - u_j)^2 = u^\top L u,
$$

- $E$ is the *Dirichlet energy*: a field is cheap to vary across an edge of small affinity and expensive across one of large affinity;
- $L$ is the graph Laplacian, $L_{ii} = d_i$, $L_{ij} = -w_{ij}$ for $i \sim j$, 0 otherwise: symmetric, positive semi-definite, with the constant vectors in its null space.

The points are *seeds*: observed values $u_k = s_k$ for $k \in S$, which condition the field
exactly. Order the voxels free first, $u = (u_F, u_S)$, so that
$L = \begin{pmatrix} L_{FF} & L_{FS} \\ L_{SF} & L_{SS} \end{pmatrix}$. The conditional of a
Gaussian is Gaussian, with mean the minimiser of $E$ over $u_F$ at fixed $u_S = s$; setting
$\partial E / \partial u_F = 0$,

$$
L_{FF}\, p_F = -L_{FS}\, s, \qquad p_S = s,
$$

and $p = (p_F, p_S)$ is what the editor computes. Row $i$ of the system reads

$$
p_i = \frac{1}{d_i} \sum_{j \sim i} w_{ij}\, p_j \quad (i \in F):
$$

every free voxel is the affinity-weighted mean of its neighbours ($p$ is *harmonic*), and the
seeds are what it is a mean of, in the end. The same $p_i$ is the probability that a random
walker started at $i$, stepping to neighbour $j$ with probability $w_{ij} / d_i$, reaches a seed
with $s = 1$ before one with $s = 0$ (the walker's exit probabilities are harmonic with the same
boundary values, and the solution is unique). This is Grady's random walker, and the reading the
probability map is given: the chance of reaching an inside point first.

Two things the model is chosen for follow from harmonicity alone, whatever the affinities:

**Maximum principle.** A harmonic function has no strict local extremum on $F$: $p_i$ is a mean
of its neighbours, so $p_i \le \max_{j \sim i} p_j$, with equality only if all neighbours are at
$p_i$. Hence, with all $w > 0$ and at least one seed at 0, *every face-connected component of
$\{p \ge t\}$, for any $t > 0$, contains an inside seed.* Proof: take a component $C$ with no
inside seed and let $i^\star$ maximise $p$ over $C$; it is free (the only seeds in $C$ are at
0 < t). All its neighbours have $p \le p_{i^\star}$, so by harmonicity all are at $p_{i^\star}
\ge t$ and in $C$. Repeating, the whole graph component of $i^\star$ -- the whole grid, which is
connected -- is at $p_{i^\star} > 0$, contradicting a seed at 0. So an inside point selects what
the affinities connect it to and *nothing lights up on its own*; the labelmap and the surface at
any threshold consist of pieces that each hold an inside point. (Cheng et al. showed that
Grady's multi-label walker, which takes an arg-max over several harmonic functions, can give a
label a piece without its seed; with two labels and a single $p$ the argument above is complete.)

**Comparison principle.** $p$ is monotone in the seeds: raising any seed value, or adding an
inside seed, can only raise $p$ everywhere; adding an outside seed can only lower it. (The
difference of the two solutions is harmonic off the seeds and non-negative on them, so by the
maximum principle non-negative everywhere.) A correction does what it says, and the clicked
voxel itself is at $p = 1$ (or 0) exactly.

A seed is one voxel (`Seeds.from_voxels`: the grid voxel nearest the clicked position). The
editor reads $p$ back onto the CT's grid trilinearly, so on a CT grid coarser than $h$ a seeded
voxel shows a little below 1 between two grid planes; on the grid itself it is exact.

## 3. What one click can do

Consider a vessel of cross-section $A$ voxels and perimeter $P$ voxels, with affinity $w_{\text{in}}$
on its internal edges and $w_{\text{wall}}$ on the edges across its wall, in a background that is
seeded at 0 within a few voxels (§4). Along the vessel the field obeys a discrete transmission
line: the axial conductance per unit length is $A\,w_{\text{in}}$, the leak to ground per unit
length $P\,w_{\text{wall}}$, and a solution $p \propto e^{-x/\lambda}$ decays over

$$
\lambda \approx \sqrt{\frac{A\,w_{\text{in}}}{P\,w_{\text{wall}}}} \;\text{ voxels} \;=\; \sqrt{\tfrac{A}{P}}\; e^{(z_{\text{in}} - z_{\text{wall}})/2}.
$$

For a coronary at 0.5 mm, $A / P \approx 1$. A single click is to carry through a 15 cm vessel
(300 voxels) at $p \ge 0.5$, so $\lambda$ must be a thousand voxels or more: a log-affinity
contrast $z_{\text{in}} - z_{\text{wall}} \ge 14$. Besides the decay there is the *capacity of a
point seed*: in three dimensions a single voxel at 1 has a finite conductance to the bulk,
about $2\,w_{\text{in}}$, and the vessel's total leak $\sum_{\text{wall}} w$ must be well below it
for $p$ to stay near 1 even next to the click. Both are statements about *ratios* of
affinities; the walker is invariant to a common scale. This is why the affinities are stored as
log-affinities and the range is wide:

$$
w_{ij} = \exp\!\big(\mathrm{clamp}(c \cdot z_{ij},\, -16,\, 16)\big),
$$

- $c$ (`affinity_scale`, 1 as trained) is the editor's one knob on the learned affinities: above 1 it sharpens the contrast between the edges the network finds strong and weak;
- the clamp bounds the dynamic range (a ratio of $e^{32}$ across the grid) for the solver's sake and gives every edge a floor, $e^{-16} \approx 10^{-7}$.

It is also why the classic walker fails here (§5): a coronary's wall in CT is not a step but a
ramp a few voxels wide (partial volume, the scanner's blur), and a ramp of small steps is a chain
of edges each keeping most of its contrast affinity. Measured on a real scan with contrast
affinities, $p$ from one click in a 3 mm vessel falls below 0.5 within a voxel or two. Telling a
ramp inside the lumen from a ramp across its wall, from context, is what the network is for.

## 4. Seeds

Three kinds, all held exactly (`rw.Seeds`):

- **inside points**, $s = 1$; **outside points**, $s = 0$. A Generate needs at least one inside point: without one nothing is selected.
- **background seeds**, $s = 0$ (`background_seeds`, on): every voxel with $I_i < \tau_{\text{HU}}$ (`seed_hu`, 0: fat is about $-100$ HU, soft tissue above 0) after $e$ erosions of that set by the face neighbourhood (`seed_erosion`, 1), so a partial-volume voxel at a vessel's edge is never seeded; a point always wins over a background seed. In the training data every lumen voxel has a voxel below 0 HU within 2 to 7 voxels, and under 0.06 % of lumen voxels are below 0 HU themselves. The background seeds are the ground the transmission line of §3 leaks to: they tie the unclicked vessels, the chambers and the tissue to 0 through many edges, so what leaks out of the clicked vessel is drained rather than spread.

Every voxel also has an edge of affinity $e^{-16}$ to a virtual node held at 0 (the *sink*). It
is negligible beside any real edge; it makes the system positive definite even with no seeds.

## 5. Affinities

**Basic** (`basic.py`): Grady's own,

$$
z_{ij} = -\frac{(I_i - I_j)^2}{2\delta^2},
$$

with $\delta$ (`contrast`, HU, 40): an edge whose voxels differ by $\delta$ keeps $e^{-1/2} = 61\,\%$
of its affinity, one of $3\delta$ keeps 1 %. The image itself is the one feature. This is the
baseline of §9.

**Learned** (`learned/`): a 3D U-Net $g_\theta$ (five resolutions, 22.6 M parameters,
normalised per voxel over its channels so windows of a volume agree where they overlap) takes
one channel, the HU volume on the fixed window $(\mathrm{clip}(\mathrm{HU}, -200, 1000) - 100) / 300$,
and gives three channels per voxel: the log-affinities of its edges to the neighbour along
$+z$, $+y$ and $+x$,

$$
z_{i,\,i+e_d} = g_\theta(I)_{i,d}, \qquad d \in \{z, y, x\}.
$$

The network does not see the points. A point is an exact observation on the field (§2), not a
hint for the network to interpret; what is learned is which voxels belong together -- the
network *learns how to learn* from a click. At session open it runs once over the whole image
in overlapping 96³ windows blended with Gaussian weights, and the log-affinities are kept as the
image's features (3 channels, float16). A Generate only seeds and solves.

## 6. Solving

The system $L_{FF}\,p_F = b$, $b = -L_{FS}\,s$ (what the seeds push into the free voxels next to
them), is symmetric positive definite and is solved on the whole grid at once with the
conjugate gradient method, the matrix-vector product written edge by edge
(`Level.apply`): for each edge, $w_{ij}(v_i - v_j)$ is added to row $i$ and subtracted from row
$j$, then $(\varepsilon_i + \text{sink})\,v_i$ is added, where $\varepsilon_i = \sum_{k \in S,\, k \sim i} w_{ik}$
is the affinity from $i$ to its seeded neighbours. The sum of these positive edge terms is
exactly symmetric and positive in float32, where forming the degree first would round the
weakest edges away (and did: an error of $4 \cdot 10^{-4}$ at the far end of a test vessel
became $10^{-6}$).

Plain conjugate gradients carry information one edge per iteration and need about two
iterations per voxel of vessel length (536 for a 300-voxel tube), so they are preconditioned
with a **multigrid cycle** (`rw.Multigrid`). Voxels are merged $2 \times 2 \times 2$ into
aggregates. With $P$ the piecewise-constant interpolation (every free voxel of an aggregate
takes the aggregate's value), the Galerkin coarse operator $P^\top A P$ has

$$
(P^\top A P)_{IJ} = -\!\!\sum_{i \in I,\, j \in J,\, i \sim j}\!\! w_{ij} \quad (I \ne J), \qquad
(P^\top A P)_{II} = \sum_{J \ne I} \lvert (P^\top A P)_{IJ} \rvert + \sum_{i \in I} \varepsilon_i :
$$

the coarse edges are the summed fine affinities between two aggregates, and the coarse extra
diagonal is the summed fine one (the fine edges inside an aggregate cancel: each appears in two
degrees and twice off the diagonal). So every level has the form of the fine one -- a 6-stencil
of affinities plus a diagonal -- and uses the same code, down to a grid of at most 2048 nodes
that is factorised and solved exactly. One V-cycle is a red-black Gauss-Seidel half-sweep pair
before the coarse correction and the same pair in reverse order after it, which makes the cycle
a symmetric positive definite operator, as a conjugate-gradient preconditioner must be. The
iteration count then hardly depends on the vessel's length: 24 iterations for the 300-voxel
tube and 34 for one twice as long, against 536 and 1093. On a $192^3$ crop of a real scan,
16 to 25 iterations.

The solver stops once the correction the preconditioner proposes is below the tolerance at every
voxel, $\max_i \lvert z_i \rvert < \tau$ with $z = M^{-1} r$ ($r$ the residual, $M^{-1}$ the
cycle), $\tau$ = `tolerance` ($10^{-4}$); with multigrid this is an estimate of the error itself,
where the plain Jacobi measure $\lvert r_i \rvert / d_i$ is local and a long vessel fools it
(`preconditioner`). The residual is recomputed from the solution every 25 iterations, so the
recurrence does not drift in float32. A Generate after a new point starts from the last
probability when the affinities have not changed, which is where a correction's solution
mostly already is. The result reports the iterations, the final measure, whether it converged,
how many pieces $\{p \ge 0.5\}$ has (one per inside point at most, by §2, when it did) and the
seeds of each kind.

Memory is about 70 bytes per voxel (the affinities twice, the diagonals, the right-hand side,
five conjugate-gradient vectors, the multigrid temporaries and a seventh more for the coarse
grids); the cap is $10^8$ voxels, a 512 × 512 × 380 scan at 0.5 mm, and a finer grid is
refused with the spacing that would fit. The labelmap is $[p_i \ge \tau_p]$ at a threshold
$\tau_p$ (0.5); the surface is the $\tau_p$ level set of $p$.

## 7. Learning the affinities

Patches are cut from the 0.5 mm cache (560 training scans), mostly centred near vessels, with
the 48 symmetries of the cube and a little HU jitter as augmentation (the affinity channels
follow the patch's axes, so the targets are made from the patch as augmented). The label is
read as binary: lumen or not.

**Edge targets** (`learned/targets.py`). For each edge $\{i, j\}$ of a patch,

$$
t_{ij} = \begin{cases} [\,y_i = y_j\,] & \text{if } y_i = 1 \text{ or } y_j = 1 \quad (\text{lumen-touching}) \\[2pt]
\exp\!\big(-(I_i - I_j)^2 / 2\delta_b^2\big) & \text{otherwise} \quad (\text{background}), \end{cases}
\qquad \delta_b = 100 \text{ HU}.
$$

The lumen-touching edges decide what a click selects: 1 between two lumen voxels, 0 across the
wall. They are a fraction of a percent of all edges. The background edges get Grady's contrast
affinity as a soft target: the walker needs the background to be a connected conductor -- what
leaks out of a vessel must reach the seeded fat and be drained, and an unclicked vessel must be
tied to that same ground -- and the contrast affinity is a sensible one for that, cut only where
tissues meet.

**Stage 1, affinities.** The loss is a binary cross-entropy on $\sigma(z_{ij})$, each class of
edges contributing its mean so that neither is drowned by the other:

$$
\mathcal L_A = \frac{1}{\lvert \mathcal E_L \rvert} \sum_{\mathcal E_L} \mathrm{BCE}\big(\sigma(z_{ij}), t_{ij}\big)
+ \frac{1}{\lvert \mathcal E_B \rvert} \sum_{\mathcal E_B} \mathrm{BCE}\big(\sigma(z_{ij}), t_{ij}\big),
$$

$\mathcal E_L$ the lumen-touching edges and $\mathcal E_B$ the background ones. The sign of
$z_{ij}$ is the same/different decision and its magnitude the confidence, so a well-trained
network already gives $z \approx \pm 7$ to $\pm 10$ on the edges it is sure of: a contrast of
14 to 20 between a lumen edge and a wall edge, what §3 asks for. 96³ patches, batch 4, 30 000
steps, AdamW at $3 \cdot 10^{-4}$ with a 1000-step warm-up and a cosine decay to a tenth.

**Stage 2, the walker.** With $\mathcal L_A$ kept, each patch (160³ now) is seeded at one lumen
voxel a user might click (`targets.random_seed`: a voxel whose face neighbours are all lumen,
when there is one), the background below 0 HU is seeded at 0 as in the editor, the walker is
solved, and its probability is scored against the lumen face-connected to the seed within the
patch, $y^{\text{tree}}$ (`targets.tree_of`):

$$
\mathcal L = \mathcal L_A + \beta\,\big[\mathrm{BCE}(p, y^{\text{tree}}) + \mathrm{Dice}_{\text{soft}}(p, y^{\text{tree}})\big],
\qquad \mathrm{Dice}_{\text{soft}} = 1 - \frac{2\sum_i p_i y_i + 1}{\sum_i p_i + \sum_i y_i + 1},
$$

$\beta$ = `--walker-weight` (1). This is the objective that matters: where the affinities let
the probability leak out of a vessel, or fail to carry it along one, the gradient says so edge
by edge. Batch 1 (2 on 32 GB), 20 000 steps at $10^{-4}$, from stage 1's best checkpoint.

**The implicit gradient** (`rw.RandomWalkerFunction`). $p_F$ solves $A(w)\,p_F = b(w)$ with
$A = L_{FF}$. Differentiating, $A\,\partial p_F = \partial b - (\partial A)\,p_F$, so for any loss
$\mathcal L(p)$ with $g = \partial \mathcal L / \partial p_F$,

$$
\frac{\partial \mathcal L}{\partial w_{ij}} = \lambda^\top \Big(\frac{\partial b}{\partial w_{ij}} - \frac{\partial A}{\partial w_{ij}}\, p_F\Big), \qquad A\,\lambda = g,
$$

using $A = A^\top$. An edge between free voxels adds $w_{ij}$ to $A_{ii}$ and $A_{jj}$ and
$-w_{ij}$ to $A_{ij}$ and $A_{ji}$, so $(\partial A\, p)_i = p_i - p_j$, $(\partial A\, p)_j = p_j - p_i$,
and $\lambda^\top \partial A\, p = (\lambda_i - \lambda_j)(p_i - p_j)$. An edge from a free $i$ to a
seed $k$ adds $w_{ik}$ to $A_{ii}$ and $w_{ik}\, s_k$ to $b_i$, giving $\lambda_i (s_k - p_i)$.
Both are the one formula

$$
\frac{\partial \mathcal L}{\partial w_{ij}} = -(\lambda_i - \lambda_j)(p_i - p_j), \qquad \lambda = 0 \text{ and } p = s \text{ on the seeds},
$$

and $\partial \mathcal L / \partial z_{ij} = w_{ij}\, \partial \mathcal L / \partial w_{ij}$ (0 where the
clamp acts). The backward pass is one more solve of the same system with the same multigrid
hierarchy, and nothing but $p$ is kept: no iteration is unrolled, which is what lets the patch
be 160³. The gradient is checked against finite differences in `tests/test_rw.py`.

**Validation and model selection.** Every 2000 steps (1000 in stage 2): the losses on fixed
validation patches, and a simulated user on whole validation scans segmented exactly as the
editor would (the network over the scan in windows, the walker on the whole grid) -- one click
at a lumen voxel, then up to four corrections at the centre of the largest error region, inside
where lumen is missed and outside where something else was taken. Reported: Dice after 1, 3 and
5 clicks against the tree the first click is in, the leak into the other tree, and the number of
pieces after the first click. `best.pt` is the checkpoint with the best single-click Dice.

## 8. Evaluation protocol

`learned/evaluate.py` runs the same simulated user on the 160 test scans' first 20 (or more),
whole, for Basic and Learned alike, and reports the mean Dice after 1, 3 and 5 clicks, the
clicks to reach Dice 0.85 and 0.90 (never reaching counts as clicks + 1), the leak into the
other tree and the number of pieces after the first click, and the seconds per click. Basic
scores near 0 from one click, as §3 predicts.

## 9. Properties worth knowing

- **Exact conditioning.** Clicks are observations, not inputs to a network: a clicked voxel is at its label, more inside clicks only raise $p$, more outside clicks only lower it, and no piece of the result is without an inside point. These hold whatever the affinities, trained or not.
- **One solve per Generate**, a few dozen conjugate-gradient iterations with the multigrid: seconds on a GPU for a whole scan at 0.5 mm, and a warm start after a new point.
- **Generality.** The network sees only the image, and the walker only affinities, so a vessel the network was never shown (the aortic root, a diseased segment, a non-coronary artery) is segmented as well as its affinities there are right: the lumen's internal edges strong and its wall weak. Where they are not, an outside point drains what leaked (a hard 0 the walker must honour) and an inside point lifts what was missed.
- **What a leak looks like.** By the comparison principle a leak is never a piece on its own: it is the clicked vessel's probability running through an open wall into a neighbour. The affinity map (the mean $z$ at each voxel) shows where the wall was open.
- **What the model cannot do.** It cannot bridge a true gap (a total occlusion with no lumen in between) from one click -- the field has nowhere to flow -- and needs a second inside click beyond it. It is invariant to the common scale of the affinities, so only ratios are learned; the clamp fixes the scale.

## 10. What this replaced, and why

The first formulation was a pairwise CRF on binary labels, $U(y) = -\sum_i a_i y_i + \sum w_{ij}[y_i \ne y_j]$,
with a network giving the unary $a$ and the couplings $w$ and seeing the clicks as input
channels, inferred by mean field with the clicks clamped and trained through unrolled sweeps.
It segmented coronaries well (Dice 0.69 to 0.93 after 1 to 5 clicks on test cubes) but failed
the requirements the editor is for: a clicked voxel did not visibly flip, because a clamp acts
on its neighbours only through $w$ against saturated unaries and is read back trilinearly; a
click's effect was whatever the network had learned to make of a click map, so it did not
generalise to vessels outside the training distribution; and mean field is conservative and
overconfident, with connectivity imposed afterwards by a rule. The walker makes the clicks exact
observations, the connectivity a theorem, and the generalisation a property of the affinities
alone; and its exact posterior has an implicit gradient, so nothing is unrolled and the training
patch can hold a good part of a tree.

## 11. Next steps

**Minimum spanning forest.** The walker solves a linear system; the power watershed limit of
the same energy (Couprie et al. 2011, $q \to \infty$) labels each voxel by the seed it reaches
along the path of largest minimum affinity -- a maximum spanning forest from the seeds, computed
by a sorted union-find in time $O(\lvert \mathcal E \rvert \log \lvert \mathcal E \rvert)$ with no
iteration. It depends on the affinities only through their *order*, so the trained network's
log-affinities can be used as they are, with no retraining: the forest's labeling is a sharper,
instantly computed alternative to thresholding $p$, and the two can be shown side by side. If
the forest is to be the thing optimised, a MALIS-style fine-tune (Funke et al. 2018: the loss on
the maximin edge of each wrongly joined or separated pair) is the training objective for it.

**Wider connectivity.** The 6-connected grid matches the labels, but the 18-connected one (face
and edge neighbours, 9 affinity channels) would let the walker step diagonally through a thin
vessel; the Galerkin coarsening keeps the 18-stencil form too.

## References

- L. Grady, "Random Walks for Image Segmentation," *IEEE TPAMI* 28(11):1768-1783, 2006. doi:10.1109/TPAMI.2006.233. (The multi-label walker with a prior: Grady, CVPR 2005.)
- C. Couprie, L. Grady, L. Najman, H. Talbot, "Power Watershed: A Unifying Graph-Based Optimization Framework," *IEEE TPAMI* 33(7), 2011.
- L. Cerrone, A. Zeng, F. A. Hamprecht, "End-to-End Learned Random Walker for Seeded Image Segmentation," *CVPR* 2019 (learned affinities with the walker's implicit gradient, in 2D).
- J. Funke et al., "Large Scale Image Segmentation with Structured Loss Based Deep Learning for Connectome Reconstruction," *IEEE TPAMI* 41(7), 2018 (MALIS).
- M.-M. Cheng et al., on the connectedness of the multi-label random walker's segments.
- Y. Boykov, M.-P. Jolly, "Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images," *ICCV* 2001 (the contrast weight).
