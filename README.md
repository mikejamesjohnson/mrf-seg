# MRFSeg: random-walker segmentation on learned affinities

A local, browser-based editor for segmenting the coronary arteries in CT. It opens a 3D image
from a file path and shows it on slices and in 3D. You click a point inside a vessel (and, if
need be, points outside what you do not want). The voxels form a graph whose edges carry
*affinities* -- how freely the label flows between two neighbours -- and the probability drawn
over the image is the random walker's: the chance that a walker started at a voxel, stepping to
a neighbour in proportion to the affinity, reaches an inside point before an outside one. That
is the exact posterior mean of a Gaussian Markov random field conditioned on your points, so a
clicked voxel is at 1 (or 0) exactly, another inside point can only raise the probability and
another outside point only lower it, and every piece of the result holds an inside point. You
can then turn the result into a labelmap and a surface mesh.

The affinities come from one of two places:

- **Basic** (works out of the box): Grady's random walker as published -- an edge's affinity
  falls with the Hounsfield step across it. Good for a bright blob with a sharp wall; weak for a
  coronary from a single point, whose walls are ramps a few voxels wide that a point seed leaks
  through within millimetres. This is the baseline.
- **Learned** (after training): a 3D U-Net whose three output channels per voxel *are* the
  log-affinities of its edges, trained from scratch on coronary CTA so that one point carries
  through the vessel it is in and no further. It needs its checkpoint; see
  [Training the network](#training-the-network).

The setup follows `pseg3d` (the Gaussian process editor this grew out of): Next.js + React +
Tailwind/shadcn for the interface, vtk.js for rendering, Zustand for state. The segmentation
runs in Python (`python/`: numpy, VTK, SciPy, PyTorch), in a small local backend the editor
talks to. The mathematics is in [`docs/MODEL.md`](docs/MODEL.md); how the pieces fit together is
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Run the editor

Needs Node.js 20.9 or newer and the Python environment. From this folder, once:

```bash
npm install
python3 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
```

Then, every time:

```bash
npm run dev
```

`npm run dev` starts the editor and the segmentation backend together, and Ctrl+C stops both.
Without `python/.venv` the editor still runs, and its Affinities panel says the backend is off.
The editor listens on port 8003 and the backend on 8004, so both can run beside pseg3d
(8001 / 8002).

Open <http://127.0.0.1:8003/editor>, type the full path of an image and press **Open**, or use
**Browse**. Or put the path in the URL, as `image=`, with the absolute path of a file on this
computer:

    http://127.0.0.1:8003/editor?image=/path/to/your/image.mha

The editor reads `.vti` (VTK image data) and MetaImage (`.mha`, or `.mhd` with its `.raw`/`.zraw`
data file). Everything is placed in LPS millimetres, using the file's origin, spacing and
orientation. The editor reads files through its own local server, which listens only on
127.0.0.1 and serves only those image types.

**Images to try.** The data stays in the pseg3d folder; give its absolute path.

- A vessel as a CT CPR shows it, with its label: `/Users/mjohnson/Documents/pseg3d/data/cpr_vessel.mha`
  (and `cpr_vessel_label.mha`), `cpr_vessel_wander.mha`, `cpr_vessel_seed11.mha`. These are
  0.1 mm images; the panel's **Resampling spacing** starts at 0.5 mm (the floor for CTs), so set
  it to 0.1-0.2 mm for them.
- A real CTA: `/Users/mjohnson/Documents/pseg3d/data/demo.mhd`.
- A bright sphere: `/Users/mjohnson/Documents/pseg3d/data/sphere.mha`. Its background is 0 HU,
  not below it, so turn **Seed the background** off (or set its threshold above 0) and give it
  an outside point or two; Basic then segments it from one inside point.
- The training scans themselves are `.npy` files in the cache and not readable by the editor;
  `python/tools/` has `extract_imagecas.py` for the NIfTI originals, which the editor cannot read
  either. To look at one, convert it to MetaImage with SimpleITK (`sitk.WriteImage(sitk.ReadImage(path), "scan.mha")`).

## Segmenting

Opening an image creates an `MRFSegmentation` for it in the backend
(`python/mrfseg/segmentation.py`). The backend starts extracting the image's features straight
away (Learned: running the network over the whole image, once), and the panel shows the
progress. The side panel follows the model: **Image**, **Affinities**, **Seeds**, **Solver**,
**Points**, **Generate**.

1. **Points.** Choose **Inside** (key **2**) or **Outside** (key **3**) and click in any slice
   view; **Navigate** (key **1**) turns placing off. An inside point goes in the lumen of a vessel
   you want; an outside point on something you do not. A Generate needs at least one inside
   point. Points can be dragged, selected and removed as in pseg3d (see the Points section of the
   panel).
2. **Generate segmentation.** Holds the inside points at 1 and the outside points at 0, and
   every voxel below the HU threshold (fat, air) at 0 as well (**Seed the background**, in the
   Seeds section: it is what drains a point's probability out of everything its vessel's wall
   does not hold in), solves the walker on the affinities, and draws two maps over the image:

   | Map             | What it is                                                                       | Drawn as                      |
   | --------------- | -------------------------------------------------------------------------------- | ----------------------------- |
   | **Probability** | p: the chance a walker from the voxel reaches an inside point first              | jet, 0 to 1 (shown at first)  |
   | **Affinity**    | the mean log-affinity of the voxel's edges: above 0 the label flows, below a wall | blue holds back, red flows   |

   Each map can be shown, hidden, faded and sampled (trilinear or nearest) on its own. The
   Solver section says how many iterations it took, whether it converged, and how many pieces
   the result has (one per inside point at most, when it did).
3. **Generate labelmap.** 1 for every voxel with probability at least **Threshold p** (0.5), 0
   elsewhere.
4. **Generate surface.** Marching cubes on the probability at the threshold. Drawn in 3D and as
   an outline on the slices.

Each step has an **Auto** box: a ticked step runs by itself whenever what it is made from
changes (a point added, moved or removed; a setting; a product's own threshold), in the
pipeline's order, and a stale segmentation is never built from. A Generate after a new point
starts the solver from the last probability, so a correction is quick.

### Settings

| Section    | Setting                | Default                                           | Meaning                                                                               |
| ---------- | ---------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Affinities | Affinities from        | Learned if its checkpoint is there, else Basic    | Grady's contrast affinities, or the network's log-affinities                          |
| Affinities | Resampling spacing, mm | the image's voxel size, at least 0.5; Learned 0.5 | The isotropic grid the features are extracted and the walker solved on               |
| Affinities | δ, contrast HU         | 40 (Basic)                                        | An edge whose voxels differ by this much in HU keeps 61 % of its affinity             |
| Affinities | Scale                  | 1 (Learned)                                       | What the network's log-affinities are multiplied by; above 1 sharpens its decisions   |
| Affinities | Compute on             | GPU if present                                    | CUDA, else the Apple GPU, else the CPU                                                |
| Seeds      | Seed the background    | on                                                | Hold every voxel below the threshold at 0, as an outside point would                  |
| Seeds      | Below, HU              | 0                                                 | The threshold: fat is about −100 HU, soft tissue above 0                              |
| Seeds      | Keep away, voxels      | 1                                                 | How far the background seeds keep from anything above the threshold (partial volume) |
| Solver     | Tolerance              | 0.0001                                            | Stop once no voxel's probability would still move by more than this                   |
| Solver     | Iterations at most     | 1000                                              | The result says whether the solver converged within these                             |
| Solver     | Preconditioner         | Multigrid                                         | A few dozen iterations whatever the vessel's length; Jacobi is cheaper per iteration and needs hundreds |

Coarser spacing is faster: the solver's time and memory go with the number of voxels (a
512 × 512 × 300 scan at 0.5 mm is about 40 M voxels and takes a few seconds on a GPU, longer on
the CPU). A grid too large for memory is refused with the spacing that would fit.

## Measuring the affinities

`python/mrfseg/learned/evaluate.py` measures a way of making affinities with a simulated user on
whole test scans, segmented exactly as the editor segments them, from one click in a coronary
tree and then up to four corrections:

```bash
cd python
.venv/bin/python -m mrfseg.learned.evaluate --cache /Users/mjohnson/Documents/pseg3d/data/casx/cache --basic --cases 20
.venv/bin/python -m mrfseg.learned.evaluate --cache /Users/mjohnson/Documents/pseg3d/data/casx/cache --model runs/walker/best.pt
```

It reports the mean Dice after 1, 3 and 5 clicks against the tree the first click is in, the
clicks it took to reach Dice 0.85 and 0.90, the leak into the other tree and the number of
pieces after the first click, and the seconds per click. Basic scores near 0 from one click on a
coronary, as the model predicts (`docs/MODEL.md`, the transmission line): it is the network's
affinities that make a single click carry.

## Training the network

`python/mrfseg/learned/` trains the Learned extractor on the 0.5 mm cache pseg3d built from
ImageCAS-X (800 scans; `<id>.image.npy`, `<id>.label.npy`, `splits.json`: see
`python/mrfseg/learned/prepare.py` for the format). The cache stays where it is; give its
absolute path. On the GPU workstation (24-32 GB), only `python/` is needed:

```bash
git clone <this repository> mrf-seg && cd mrf-seg/python
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch with CUDA
./train.sh sanity   /path/to/pseg3d/data/casx/cache   # ~10 min: 300 steps of stage 1 and a validation; checks memory and that the loss falls
./train.sh affinity /path/to/pseg3d/data/casx/cache   # stage 1: 30 000 steps on 96^3 patches, validating every 2000; rerun the same line to resume
./train.sh walker   /path/to/pseg3d/data/casx/cache   # stage 2: 20 000 steps on 160^3 patches from stage 1's best; rerun to resume
./train.sh eval     /path/to/pseg3d/data/casx/cache   # the simulated user on 20 whole test scans, Learned against Basic
```

`PIECE_MINUTES=120 ./train.sh affinity …` ends a piece after two hours and the same command
carries on from where it stopped; `EXTRA="--patch 128"` when the walker stage is short of
memory. The script is a thin wrapper over `python -m mrfseg.learned.train`, whose `--help`
lists everything. Then, back on the Mac:

```bash
scp workstation:mrf-seg/python/runs/walker/best.pt ~/.cache/mrfseg/learned.pt   # or export MRFSEG_LEARNED_WEIGHTS=/path/to/best.pt
npm run dev
```

The two stages (`python/mrfseg/learned/__init__.py`):

1. **Affinities.** Every edge of a patch gets a target: where either voxel is lumen, 1 if both
   are and 0 across the wall; between two background voxels, the contrast affinity with
   δ = 100 HU, which keeps the background a connected conductor to the seeded fat. The loss is a
   binary cross-entropy on sigmoid(z), the lumen-touching edges (a fraction of a percent of all
   edges) and the rest weighed equally.
2. **The walker.** With that loss kept, each (larger) patch is seeded at one lumen voxel a user
   might click, the background below 0 HU is seeded as outside, the walker is solved, and its
   probability is scored against the lumen connected to the seed with cross-entropy and soft
   Dice. The gradient reaches the network through the solution *implicitly* -- one more linear
   solve, no iterations unrolled -- so the patch can be 160³. This is the objective that matters:
   where the affinities let the probability leak out of a vessel, or fail to carry it along one,
   the gradient says so edge by edge.

Both stages validate every so often with a simulated user on whole validation scans (Dice after
1 / 3 / 5 clicks, the leak into the other tree, the number of pieces), and `best.pt` is the
checkpoint with the best single-click Dice (what the editor loads); `last.pt` is the whole state
of the run -- network, optimiser, loss scaler, step, best score -- written at every validation
(`--checkpoint-every` to change) and whenever a piece ends, with an atomic rename, so a crash
loses at most one validation interval. `./train.sh affinity` and `walker` resume from `last.pt`
by themselves when there is one (`--resume`); `--init best.pt` starts a fresh run from a
checkpoint's weights instead (how stage 2 starts). Checkpoints, logs and runs are ignored by git
(`runs/`, `*.pt`). On CUDA the network runs in bf16 (Ampere and later) or fp16 with loss
scaling; the solver and the losses run in fp32. A Mac's GPU works too, slowly (`--device mps`).

Once `MRFSEG_LEARNED_WEIGHTS` points at a checkpoint (or it sits at `~/.cache/mrfseg/learned.pt`),
the editor offers **Learned** and chooses it by default. Its features are the network's
log-affinities, extracted once per image in overlapping 96³ windows; a Generate only seeds and
solves.

`npm run backend` runs the backend on its own (`python -m mrfseg.server`, port 8004), for
working on it.

## Layout

```
src/app/                  pages; api/file serves local files, api/browse lists folders, api/mrf forwards to the backend
src/components/           toolbar (with Browse), status bar, side panels, viewport panes
src/lib/scene/            the editor's state: datasets, views, points (store slices), data registry
src/lib/io/               image file formats, into LPS mm
src/lib/segmentation/     the backend client, and Generate, the labelmap and the surface on top of it
src/lib/viewer/           vtk.js rendering: one ViewportController per pane
src/lib/editor/           commands across all of those: opening an image, the debug hook
python/mrfseg/            the core: MRFSegmentation, settings, features, the walker, surface, image reader
python/mrfseg/rw.py       the random walker: the model, its seeds, the multigrid-preconditioned solver, the implicit gradient
python/mrfseg/basic.py    the Basic affinities: Grady's contrast weights from the image's HU
python/mrfseg/settings.py MRFSettings: defaults and bounds, mirrored by the editor (tests/settings.test.ts)
python/mrfseg/extractors.py  the feature extractors to choose from, and how each makes the affinities
python/mrfseg/learned/    the network, and the pipeline that trains and measures it (targets, data, train, evaluate)
python/mrfseg/windows.py  running a network over a volume in overlapping, blended windows
python/mrfseg/server/     the local backend: one MRFSegmentation per open image
python/mrfseg/phantoms.py synthetic test volumes: a vessel as a CT CPR shows it, with its label
python/tests/             tests of the core and the backend
python/tools/             make_sphere_image.py (the sphere test image), extract_imagecas.py (the Kaggle split zips)
tests/                    unit tests of the editor's modules and store (vitest)
scripts/with-backend.mjs  runs the editor and the backend together (npm run dev / start)
```

## Tests

`npm test` runs both suites: `npm run test:web` (the editor's unit tests in `tests/`, with
vitest) and `npm run test:python` (the core and the backend). `npm run typecheck`,
`npm run lint` and `npm run build` check the rest of the editor.

## VS Code

`.vscode/settings.json` points the Python extension at `python/.venv`, so set up the virtual
environment first. The core tests then show up in the Testing panel. TypeScript needs nothing
beyond `npm install`.
