# MRFSeg: conditional random field segmentation

A local, browser-based editor for segmenting the coronary arteries in CT with a Markov random
field. It opens a 3D image from a file path and shows it on slices and in 3D. You click points
inside and outside the vessel. A conditional random field (CRF) over the voxels -- a unary
potential per voxel saying how much the evidence favours lumen, and a pairwise potential per
pair of neighbours saying how much a label change between them costs -- is inferred with your
points clamped, and the probability that each voxel is lumen is drawn over the image. You can
then turn the result into a labelmap and a surface mesh.

The potentials come from one of two places:

- **Basic** (works out of the box): the image's brightness at several scales, with potentials
  made by hand from the points: a kernel-density likelihood ratio for the unary, and a
  contrast-sensitive coupling in Hounsfield units. This is what to start with.
- **Learned** (after training): a 3D U-Net whose four output channels per voxel *are* the
  potentials, trained from scratch on coronary CTA with coronary label maps through the same
  mean-field inference the editor runs. It needs its checkpoint; see
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
Without `python/.venv` the editor still runs, and its Feature extraction panel says the backend
is off. The editor listens on port 8003 and the backend on 8004, so both can run beside pseg3d
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
- A bright sphere: `/Users/mjohnson/Documents/pseg3d/data/sphere.mha`. It is a third of its
  image, not the rare structure the Basic unary expects of coronary lumen: raise **a₀, prior
  logit** to about 2 and **ℓ features** to 1 for it.
- The training scans themselves are `.npy` files in the cache and not readable by the editor;
  `python/tools/` has `extract_imagecas.py` for the NIfTI originals, which the editor cannot read
  either. To look at one, convert it to MetaImage with SimpleITK (`sitk.WriteImage(sitk.ReadImage(path), "scan.mha")`).

## Segmenting

Opening an image creates an `MRFSegmentation` for it in the backend
(`python/mrfseg/segmentation.py`). The backend starts extracting the image's features straight
away, and the panel shows the progress. The side panel follows the model: **Image**, **Feature
extraction**, **Unary potential**, **Pairwise potential**, **Inference**, **Points**, **Generate**.

1. **Points.** Choose **Inside** (key **2**) or **Outside** (key **3**) and click in any slice
   view; **Navigate** (key **1**) turns placing off. Basic needs at least one inside point;
   Learned needs none. Points can be dragged, selected and removed as in pseg3d (see the Points
   section of the panel).
2. **Generate segmentation.** Builds the potentials from the features and the points, clamps the
   points to their labels, runs the mean-field sweeps, and draws three maps over the image:

   | Map             | What it is                                                            | Drawn as                          |
   | --------------- | --------------------------------------------------------------------- | --------------------------------- |
   | **Probability** | p(y = 1): the mean-field marginal, the chance a voxel is lumen        | jet, 0 to 1 (shown at first)      |
   | **Unary logit** | a: the evidence alone, before the coupling has its say; positive is lumen | blue inside, red outside      |
   | **Coupling**    | the mean coupling of the edges at each voxel: low where a label may change cheaply | dark to bright        |

   Each map can be shown, hidden, faded and sampled (trilinear or nearest) on its own. The
   Inference section says how many sweeps it took and whether it converged.
3. **Generate labelmap.** 1 for every voxel with probability at least **Threshold p** (0.5), 0
   elsewhere. **Connected to the inside points** (on by default) keeps only the connected
   components an inside point touches: a pairwise CRF cannot promise one connected vessel tree,
   so islands of lumen-like voxels are labelled too, and this is where they go.
4. **Generate surface.** Marching cubes on the probability at the threshold, or on the unary
   logit's zero level (the evidence alone). Drawn in 3D and as an outline on the slices.

Each step has an **Auto** box: a ticked step runs by itself whenever what it is made from
changes (a point added, moved or removed; a setting; a product's own threshold), in the
pipeline's order, and a stale segmentation is never built from.

### Settings

| Section            | Setting                 | Default                                                | Meaning                                                                                  |
| ------------------ | ----------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Feature extraction | Extractor               | Learned if its checkpoint is there, else Basic         | What makes the features, and so the potentials                                           |
| Feature extraction | Resampling spacing, mm  | the image's voxel size, at least 0.5; Learned 0.5      | The isotropic grid the features are extracted and the CRF inferred on                   |
| Feature extraction | Scales, Edges, Shape    | 4, off, off (Basic)                                    | The blurs, and their edge strength and Hessian eigenvalues                               |
| Feature extraction | Compute on              | GPU if present                                         | CUDA, else the Apple GPU, else the CPU                                                   |
| Unary potential    | ℓ features              | 0.5 (Basic)                                            | Kernel width in feature space, in standard deviations of the image's features            |
| Unary potential    | ℓ position, mm          | 20 × the image's largest voxel (Basic)                 | How far a point's evidence reaches                                                       |
| Unary potential    | a₀, prior logit         | −2 (Basic)                                             | Added to every logit: how much the evidence for lumen must outweigh the evidence against |
| Unary potential    | β, evidence scale       | 1 (Basic)                                              | What the evidence is multiplied by, against the coupling                                 |
| Unary potential    | Shift                   | 0 (Learned)                                            | Added to the network's logits                                                            |
| Pairwise potential | λ, coupling             | 1                                                      | The coupling of a contrast-free edge (Basic), or a factor on the network's (Learned)     |
| Pairwise potential | δ, contrast HU          | 100 (Basic)                                            | An edge with this HU step keeps 61 % of the coupling                                     |
| Pairwise potential | Floor                   | 0.1 (Basic)                                            | The share of the coupling every edge keeps                                               |
| Inference          | Sweeps                  | 50 (Learned: as trained)                               | Mean-field sweeps; stops early once nothing changes by more than 0.001                   |

Coarser spacing is faster: the inference's time and memory go with the number of voxels (a
512 × 512 × 300 scan at 0.5 mm is about 40 M voxels and takes a few seconds on an Apple GPU,
longer on the CPU). A grid too large for memory is refused with the spacing that would fit.

## Measuring the potentials

`python/mrfseg/learned/evaluate.py` measures a way of making potentials with a simulated user on
cubes cut from the cache's test scans, segmented exactly as the editor segments:

```bash
cd python
.venv/bin/python -m mrfseg.learned.evaluate --cache /Users/mjohnson/Documents/pseg3d/data/casx/cache --basic --cases 20
.venv/bin/python -m mrfseg.learned.evaluate --cache /Users/mjohnson/Documents/pseg3d/data/casx/cache --model runs/first/best.pt
```

It reports the mean Dice after 1, 3, 5 and 10 clicks, the clicks it took to reach Dice 0.85
and 0.90, and the leak into other vessels. With the Basic defaults on 8 test cubes of 96³ at
0.5 mm the Dice was 0.42 / 0.53 / 0.58 after 1 / 3 / 5 clicks with no leak: the handcrafted
potentials are precise but short of recall, which is what the network is for.

## Training the network

`python/mrfseg/learned/` trains the Learned extractor on the 0.5 mm cache pseg3d built from
ImageCAS-X (800 scans; `<id>.image.npy`, `<id>.label.npy`, `splits.json`: see
`python/mrfseg/learned/prepare.py` for the format). The cache stays where it is; give its
absolute path. On the GPU workstation (24 GB), only `python/` is needed:

```bash
git clone <this repository> mrf-seg && cd mrf-seg/python
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch with CUDA
./train.sh sanity /path/to/pseg3d/data/casx/cache    # ~10 min: 300 steps and a validation; checks memory and that the loss falls
./train.sh full   /path/to/pseg3d/data/casx/cache    # 40 000 steps, validating every 2000; rerun the same line to resume
./train.sh eval   /path/to/pseg3d/data/casx/cache    # the simulated user on 20 test scans, Learned against Basic
```

`FULL_PIECE_MINUTES=120 ./train.sh full …` ends a piece after two hours and the same command
carries on from where it stopped; `EXTRA="--checkpoint-sweeps"` (or `--batch 2 --accumulate 2`)
when short of memory. The script is a thin wrapper over `python -m mrfseg.learned.train`, whose
`--help` lists everything. Then, back on the Mac:

```bash
scp workstation:mrf-seg/python/runs/first/best.pt ~/.cache/mrfseg/learned.pt   # or export MRFSEG_LEARNED_WEIGHTS=/path/to/best.pt
npm run dev
```

`train` draws 96³ patches centred near vessels, simulates a user's clicks on each (none to 3
inside, none to 4 outside, mostly near the wall), runs the network, then 20 mean-field sweeps
*through which the loss is back-propagated*, and scores the marginals against the lumen label
with cross-entropy plus a soft Dice; a lighter cross-entropy on the unary alone keeps it a
usable classifier by itself. It prints the loss and the Dice on the training patches, validates
every 2000 steps with a simulated user on sub-volumes of validation scans (Dice with 0 / 1 / 3 /
5 clicks, and the leak into other vessels), and keeps `best.pt` (by that Dice) and `last.pt`
(with the optimiser). The checkpoint records the settings the editor should start from with it
(the sweeps it was trained through). On CUDA it runs in bf16 (Ampere and later) or fp16 with
loss scaling; a Mac's GPU works too, slowly (`--device mps`). Checkpoints, logs and runs are
ignored by git (`runs/`, `*.pt`).

Once `MRFSEG_LEARNED_WEIGHTS` points at a checkpoint (or it sits at `~/.cache/mrfseg/learned.pt`),
the editor offers **Learned** and chooses it by default. Its features are the network's raw
potentials, extracted once per image in overlapping 96³ windows; a Generate is then the
inference alone, and the points only clamp.

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
python/mrfseg/            the core: MRFSegmentation, settings, features, the CRF, surface, image reader
python/mrfseg/mrf.py      the model and its mean-field inference (also batched and differentiable, for training)
python/mrfseg/handcrafted.py  the Basic potentials: kernel-density unary from the points, contrast-sensitive coupling
python/mrfseg/settings.py MRFSettings: defaults and bounds, mirrored by the editor (tests/settings.test.ts)
python/mrfseg/extractors.py  the feature extractors to choose from, and how each makes the potentials
python/mrfseg/basic.py    the default feature extractor: multi-scale blurs (+ edges, shape)
python/mrfseg/learned/    the network, and the pipeline that trains and measures it (data, clicks, train, evaluate)
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
