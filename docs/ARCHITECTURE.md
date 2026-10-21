# Architecture

MRFSeg is two programs on one computer: a Next.js editor in the browser, and a Python backend
holding the conditional random field. The browser only ever talks to the editor's own server,
which reads image files for it (`/api/file`) and forwards segmentation requests (`/api/mrf`).
The model itself is written out in [`MODEL.md`](MODEL.md).

```
browser ── /api/file ──> local disk
   │
   └────── /api/mrf ────> python -m mrfseg.server ──> MRFSegmentation (one per open image)
```

## Rules

These hold everywhere; most bugs here would come from breaking one.

1. **World space is LPS millimetres.** Every actor is already in it. Only the loaders
   (`lib/io/`, `mrfseg/image_io.py`) may correct geometry, and the two readers must agree: the
   editor checks the geometry the backend read against its own.
2. **The store holds plain data only.** vtk.js objects live in `scene/dataRegistry.ts`, keyed by
   dataset id; pipelines live in `viewer/`. Nothing vtk.js goes into React or Zustand state.
3. **A dataset is what you have; a scene node is how it is drawn.** A dataset's nodes are fixed
   by its kind (`NODE_TYPES_FOR_KIND`). A replaced dataset keeps its id and bumps `version`, and
   anything that caches it keys on `id:version` (`inputKey`).
4. **The pane owns the cutting plane.** A slice node has no orientation; each slice view's
   plane (`scene/views.ts`) decides where every volume is cut, so one node serves every pane.
5. **One place knows the wire format** on each side: `lib/segmentation/backend.ts` and
   `mrfseg/server/`. Everything else uses camelCase app types.
6. **What is drawn over the image has a layer** (`viewer/layers.ts`): image, map, labelmap,
   outline, each drawn a fixed fraction of a millimetre nearer the camera than the one below,
   by moving its *actor* and never the plane it samples. Coplanar geometry would otherwise fight
   over depth; a moved sampling plane would show other voxels (a layer or two off, on a fine
   grid).
7. **Each side writes down the shared things once, and a test checks the other agrees.**
   Settings, their bounds and defaults; the extractors' names; the maps and their compact types.
   `tests/settings.test.ts` reads the Python source; `python/tests/test_server.py` checks the
   wire shapes the editor's types expect.

## The editor (`src/`)

Layers, each depending only on those above it (the store, which composes every slice, is the
one exception: `scene/store.ts` imports `segmentation/state.ts` for its slice):

| Layer               | What it is                                                                    |
| ------------------- | ----------------------------------------------------------------------------- |
| `lib/world.ts`      | vectors, bounds, voxel grids; `lib/color.ts`                                  |
| `lib/scene/`        | the state: types, the store and its slices, the data registry, histograms     |
| `lib/io/`           | file formats to datasets                                                      |
| `lib/segmentation/` | the backend client; connecting, Generate, its products and auto update        |
| `lib/viewer/`       | vtk.js: a `ViewportController` per pane, representations, the point layer     |
| `lib/editor/`       | commands across layers: opening an image, the `window.__mrfseg` debug hook      |
| `components/`       | React: panels read the store and call its actions or the commands            |

**The store** (`scene/store.ts`) is one Zustand store made of slices, each in the file that owns
the concept:

| Slice                     | State                                                          |
| ------------------------- | -------------------------------------------------------------- |
| `scene/datasets.ts`       | datasets, nodes, world bounds, load status                     |
| `scene/views.ts`          | each slice view's plane, layout, what each pane shows          |
| `scene/points.ts`         | the labelled points, the selection, the place mode             |
| `segmentation/state.ts`   | MRF settings, the session and its progress, its products        |

Pure questions about state (`currentImage`, `scrollStep`, `planesForThreeD`, `isBusy`,
`staleReasons`, `productStale`) sit beside the slice they read, and the panels and auto update
ask them rather than deciding for themselves. An action may read the whole state, but writes
only its own slice; what spans slices is a command, like `editor/openImage.ts`.

**The segmentation layer** (`lib/segmentation/`):

| Module          | What it is                                                                      |
| --------------- | ------------------------------------------------------------------------------- |
| `settings.ts`   | `MRFSettings` and the panel's table of them (mirrors `mrfseg/settings.py`), grouped as the model is: features, unary, pairwise, inference; an extractor's starting values |
| `state.ts`      | the slice, and the questions asked of it                                        |
| `maps.ts`       | the three maps (probability, unary logit, coupling): label, compact type, how each is first drawn and its legend |
| `backend.ts`    | the wire format: requests to `/api/mrf`, binary layouts                          |
| `flow.ts`       | what the async flows share: the current-image guard, polling, a lost session    |
| `session.ts`    | opening the image's session with the page, and keeping it in step with it       |
| `extractors.ts` | what the backend says about the extractors; choosing one loads its settings     |
| `generate.ts`   | Generate: post, poll, fetch and store the maps                                  |
| `products.ts`   | the labelmap and the surface, and `PRODUCTS`, the one list of them              |
| `autoUpdate.ts` | the reconciler that reruns whatever a change makes out of date                  |

**Rendering.** Each pane mounts a `ViewportController`, which subscribes to the store outside
React and diffs the nodes against its representations (`viewer/representations/`, one factory
per node type). It syncs at most once per frame, and only when something *its role* draws from
changed (`viewer/redraw.ts`): a job's progress renders nothing, and another slice pane
scrolling redraws only the 3D pane, which shows every plane. A pane's role decides what it
draws (`showsNodeType`), its camera (`viewer/cameras.ts`) and its wheel (`viewer/wheel.ts`). The
point layer is a 2D canvas over each pane that draws, picks and drags points in screen space.

- A slice pane has a second renderer over the first, sharing its camera, for the layers in
  `FOREGROUND`. vtk.js blends translucent actors order-independently, which would mix a labelmap
  with the map under it; drawn afterwards, it goes over it. The 3D pane has one renderer, since
  there depth decides.
- Maximising a pane or restoring the layout mounts new panes. Each pane's camera is kept per
  role, so the view survives that, and a pane that goes clears its extent and frees its WebGL
  context (vtk.js frees neither the GPU nor its interactor on delete). A new image re-frames
  every pane, even one on the same grid as the last: the fit key names the image.
- A slice in a standard orientation sits on a layer of voxel centres (`onVoxelLayer`), so it
  shows one layer, as a labelmap drawn voxel by voxel does. Scrolling, orienting, resetting and
  jumping to a point settle there; the reformat handle does not, as it moves the plane a little
  at a time.
- Where a camera points is not in the store: panning and zooming would write it every frame,
  and nothing else reads it. Camera commands go to the mounted controllers
  (`viewer/activeViewport.ts`), and the two the store does carry are **events**, which a pane
  acts on once: `sliceFits`, a counter per view that asks it to frame the volume again, and
  `sliceSync`, Slicer's Shift-hover. A hover over a slice pane (`viewer/shiftHover.ts`) picks
  the point under the pointer on the pane's plane (`viewer/picking.ts`, shared with the point
  layer) and `syncSlicesTo` moves every other view's plane through it and records the event in
  the same update, so a pane centring on the point reads its plane as already there. The
  hovered pane is named in the event and ignores it: the point is on its plane, and panning it
  would run the image away from the pointer. While the hover lasts, the other slice panes draw
  a crosshair at the point (the point layer, being each pane's screen-space canvas); it is
  cleared by `endSliceSync` when Shift is released or the pointer leaves, and the views stay.

## The backend (`python/`)

| Module                    | What it is                                                        |
| ------------------------- | ----------------------------------------------------------------- |
| `mrfseg/settings.py`      | `MRFSettings`, its `BOUNDS` and validation (mirrored by the editor) |
| `mrfseg/segmentation.py`  | `MRFSegmentation`: features (cached), generate, products          |
| `mrfseg/mrf.py`           | the CRF (`Potentials`, `Clamps`) and its mean-field inference, plus the batched, differentiable version training uses |
| `mrfseg/handcrafted.py`   | the Basic potentials: kernel-density unary from the points against the image's background, contrast-sensitive coupling |
| `mrfseg/features.py`      | the feature grid and its memory cap, resampling, feature lookup, the HU copy the coupling reads |
| `mrfseg/extractors.py`    | the table of extractors: how each is made, what settings it reads, how it makes the potentials |
| `mrfseg/basic.py`         | the default extractor: multi-scale blurs, optional edges, shape   |
| `mrfseg/learned/`         | the network, its extractor, and everything that trains and measures it (see below) |
| `mrfseg/windows.py`       | running a network over a volume in overlapping, blended windows   |
| `mrfseg/surface.py`       | marching cubes on a map, in world space                           |
| `mrfseg/grid.py`          | vtkImageData <-> numpy, and index <-> world                       |
| `mrfseg/image_io.py`      | reading images (as the editor does) and writing MetaImage         |
| `mrfseg/phantoms.py`      | synthetic test volumes with known answers: a vessel in a CT CPR   |
| `mrfseg/errors.py`        | `InputError`, and what counts as a number                         |
| `mrfseg/compute.py`       | the torch device, and progress callbacks                          |
| `mrfseg/server/`          | HTTP (`app.py`), sessions and background jobs, binary encodings   |

**Features.** The settings name the extractor (`feature_extractor`), and `extractors.EXTRACTORS`
says how to make it, which settings it reads, and how its features become the CRF's potentials;
`MRFSegmentation` uses that unless it was given an extractor (and a potentials maker), as the
tests and the training's validation do. The features are extracted again exactly when
`extractors.features_key` changes: the extractor, its grid, and what it reads. Either runs once,
on one isotropic grid (`feature_spacing`): the image's finest spacing floored at 0.5 mm by default
(or the spacing the network was trained on), doubled until the grid fits the features' memory cap
and the inference's (`segmentation.check_grids`). Resampling is one small matrix per axis (a
Gaussian, then linear interpolation), applied a few CT slices at a time, and a float16 copy of the
resampled HU is kept beside the features for the contrast-sensitive coupling.

- **Basic** (`basic.py`, the default): blurs at sigma = 1, 2, 4... grid voxels, with each blur's
  edge strength and Hessian eigenvalues if asked, each standardised over the image and all scaled
  by 1/sqrt(C). Its potentials are handcrafted (`handcrafted.py`): a Gaussian is fitted to a
  random sample of the image's features (background), and each voxel's unary logit is the log
  ratio of the inside points' kernel density (Mahalanobis in the background's metric, times a
  spatial kernel) to the background's, or to the background's mixed with the outside points'.
  The coupling is Boykov & Jolly's in HU. Needs an inside point.
- **Learned** (`learned/`): our own 3D U-Net (22.6 M parameters, four channels per voxel: the
  unary logit and the raw couplings along +z, +y, +x), trained with `learned/train.py` on
  coronary CTA with coronary label maps *through the mean-field inference* (`mrf.mean_field_batched`,
  with simulated clicks clamped), so that the marginals reproduce the label map. It takes HU on
  a fixed window at 0.5 mm, normalises per voxel over its channels (`model.ChannelNorm`) rather
  than over the patch, and runs in windows of the training patch size (96³), half overlap,
  Gaussian blending (`windows.py`), with empty click maps: the automatic potentials, kept as the
  image's features. A Generate with points re-runs it in windows centred on the points, with the
  click maps as input (`learned.network_input`), and blends the result into those
  (`Learned.refine`); the potentials are the unary plus `unary_shift`, and `coupling` times the
  softplus of the raw couplings (`model.PotentialNet.potentials`). The checkpoint is
  `$MRFSEG_LEARNED_WEIGHTS` or `~/.cache/mrfseg/learned.pt`, loaded once per version of the
  file; it carries its config, the spacing, and the settings the editor should start from
  (`setting_defaults`: the sweeps it was trained through). Choosing it without a checkpoint is
  an `InputError` with the training command, and `GET /extractors` says so beforehand, so the
  editor greys it out. The training side -- `data.py` (patches, labels, simulated clicks),
  `clicks.py` (the simulated user, shared with the evaluation and the training), `train.py`,
  `evaluate.py` (a simulated user on test cubes, for both ways of making potentials) and
  `prepare.py` (NIfTI pairs -> a 0.5 mm cache) -- is documented in the package and the README;
  the editor never imports it.

**The inference** (`mrf.py`) is checkerboard mean field: the field h_i = sum_j w_ij (2 m_j - 1)
is computed for the whole volume with six shifted multiply-adds, one parity is updated to
sigmoid(a + h), the clamps are reapplied, then the other parity; a sweep is both. It stops early
once no marginal changes by more than 10^-3. The memory is about 34 bytes per voxel
(`mrf.BYTES_PER_VOXEL`), capped at 100 M voxels (`MAX_MRF_BYTES`). The maps -- the marginal, the
unary and the mean coupling at each voxel -- are read back onto the CT's grid a slice at a time
(`mrf.resample_to`).

**Generate** (`MRFSegmentation.generate_segmentation`) extracts the features if they are not
there, builds the potentials from them and the points, clamps the points' voxels, runs the
sweeps, keeps what the inside points connect to if `connected` is on (`select_connected`: the
face-connected components of {p >= 0.5} holding an inside point; p is set to 0 on the others),
and reads the maps onto the CT's grid: the values the user sees are the values used, and the
same points and settings give the same maps. There is no Optimize: the handcrafted potentials
have no fit, and the network's is its training.

**The server** wraps the core, which has no HTTP or UI code. Feature extraction and Generate run
on background threads (one Generate at a time), and the editor polls the session for progress
and results. What can be checked is checked at the edge, before anything starts: a request with
bad points or a grid too large for memory is a 400 and touches nothing. The last result, and the
products built from it, stay until a Generate replaces them (the old maps go only as the new
ones are about to be read), so a Generate that fails leaves them as they were; the features Task
is kept true to the features throughout, running again when a Generate needs new ones. A request
the session cannot serve yet is a 409 (`NotReady`). Closing a session (the image is replaced,
the page goes, or a newer session evicts it) cancels its work at the next progress step.

**Who may call.** Both servers bind to 127.0.0.1 and refuse a non-loopback `Host` (DNS
rebinding). The editor's routes (`lib/server/local.ts`) serve only its own page, by
`Sec-Fetch-Site` and `Origin`; the backend serves only the editor's server, so refuses any
`Origin`. A POST must be JSON on both, which another site cannot send without a preflight.

## Flows

- **Open an image** (`editor/openImage.ts`, typed, from `?image=`, or picked in the Browse
  dialog, which lists folders through `/api/browse`): load it in the browser, replace what was
  open, reset the views, points and segmentation, then open a backend session, which reads the
  same file and starts extracting features. Before the session opens, the backend is asked about
  the extractors once (`segmentation/extractors.ts`), so the first session already uses Learned
  when its checkpoint is there; opening fills in the settings the image chooses (`IMAGE_DERIVED`,
  but not a value the extractor brought, such as Learned's ℓ position).
- **Choose an extractor** (`extractors.ts`): `chooseExtractor` writes `settingsForExtractor` into
  the settings -- the extractor, the grid left to the image, and the editor's defaults for the
  potentials and the inference overlaid with the extractor's own values -- and records it in
  `defaultsFrom`; `resetPotentials` writes that part again without touching the features.
- **Generate** (`segmentation/generate.ts`): connect if there is no session, *then* copy the
  settings (so they hold the image's choices), post the points, poll the job, then fetch the
  three maps (compact int16/uint8) and store them as map datasets on the image's grid. The job
  stays "running" (stage `maps`) until the maps and `generated` are stored together, so nothing
  can start another Generate in between. `generated` records the session it came from: a new
  session has no segmentation, so the maps are out of date under it (`staleReasons`). Only the
  latest Generate asked for may write its result; a session that changes under a poll ends it
  with an error rather than a job left "running".
- **Auto** (`segmentation/autoUpdate.ts`): a reconciler. A store subscription, installed by the
  editor shell, watches what the outputs are made from (`pointsVersion`, the MRF settings, the
  session, `autoUpdate`, and each product's own settings from `PRODUCTS[].inputs`) and asks for
  a run when any of it changes while any step's Auto box is ticked. A run waits until nothing is
  running (`isBusy`), no pointer button is held (a drag moves a point on every pointer move) and
  nothing has changed for 250 ms; then, in the pipeline's order and only for ticked steps, it
  Generates if `needsGenerate`, and builds each product for which `PRODUCTS[].stale` -- never
  from a stale segmentation, so an unticked step holds up the ones after it. Requests during a
  run become one more run, which also skips the builds of the run it follows. Outputs changing
  never ask for a run, so a failed run is not repeated until an input changes. Number fields
  commit on leaving, so typing changes no input.
- **Products** (`segmentation/products.ts`): built from the last segmentation on request, each
  through `build`, which reports in the product's slice of state (`Product`), refuses when there
  is no session or no segmentation under it, and stores the result only if its image is still
  open. Every step's button is off while anything runs (`isBusy`).
  - **Labelmap**: the backend thresholds p on its exact maps; the labels come back as uint8 and
    are stored as a labelmap dataset, drawn like a map but in one colour, without interpolation.
  - **Surface**: the backend runs marching cubes on a map; the mesh comes back as one binary
    buffer and is stored as a mesh dataset. In 3D the mesh is drawn; in a slice view, where the
    view's plane cuts it (`meshPlaneSegments`).

## Extending

- **An MRF setting**: a field of `MRFSettings` in `mrfseg/settings.py`, with its validation (a
  bound in `BOUNDS`, or a place in `FLAGS`), and in `segmentation/settings.ts` a field of
  `MRFSettings`, a default and a `SETTING_GROUPS` entry; the panel draws its row from that table.
  `tests/settings.test.ts` checks the two sides agree, bounds included. A setting one extractor
  reads goes in that extractor's `reads` in `extractors.py`, and gets a `when` in the panel's
  table; one of the potentials' or the inference's goes in `POTENTIAL_KEYS` too, so an extractor
  may bring its own value for it.
- **A feature extractor**: a class with `channels`, `unit_vectors` and
  `__call__(volume, device, progress) -> (C, z, y, x)` (`FeatureExtractor`; `input = "hu"` and
  `native_spacing` if it wants them); a potentials maker (`extractors.PotentialsMaker`); an entry
  in `extractors.EXTRACTORS` saying how to make it from the settings, which settings it reads and
  whether it needs an inside point; and its name in `FeatureExtractorName` and `EXTRACTOR_LABELS`
  in `settings.ts`, from which the panel's dropdown is built. It is run once, on the whole
  feature grid; a network does that in windows with `windows.py`.
- **A map**: a field of `Segmentation`, filled in `MRFSegmentation._maps`, and an entry in
  `encoding.MAPS` (Python); then `MapQuantity` in `scene/types.ts` and an entry in
  `segmentation/maps.ts` (label, compact type, first appearance, colour span and legend).
- **A product**: endpoints in `mrfseg/server/app.py` and a `Session.build_*` in `sessions.py`; in
  the editor its state in `state.ts` (a `Product`, a `ProductKey`, a `productStale` rule and a
  default in `autoUpdate`), a build function and `PRODUCTS` entry in `products.ts`, and a block
  in the Generate panel with a `StepButton` (the button and its Auto box, `auto-update-panel.tsx`)
  and `ProductStatus`, which take the key. Auto update loops over `PRODUCTS`.
- **An extractor's own settings**: a `describe` in its `extractors.EXTRACTORS` entry, giving
  `settings` by their names; `settingsForExtractor` loads them when the extractor is chosen.
- **A way to draw something**: a node type in `scene/types.ts` (with its `NODE_TYPES_FOR_KIND`
  and `NODE_TYPE_LABELS` entries and a `defaultNode` case in `scene/datasets.ts`), a factory in
  `viewer/representations/`, and which panes draw it in `viewer/redraw.ts`. If it is drawn over
  the image, give it a layer in `viewer/layers.ts`; if it reads store state no representation
  reads yet, add that to `redraw.ts`.
- **An endpoint**: a route in `mrfseg/server/app.py` (`ROUTES`), state in `sessions.py`, and a
  function in `backend.ts`.

## Tests

- `npm run test:python`: the core and the backend, over real HTTP with a stub extractor
  (`tests/helpers.StubExtractor`, which records what it was given and can be made to fail).
- `npm run test:web`: the editor's pure modules and store, in Node (`tests/`, vitest): the
  segmentation state's questions, auto update against mocked Generate and builds, what makes
  each pane redraw, and the parity checks against the Python source. `npm test` runs both.
- `npm run typecheck`, `npm run lint`, `npm run build`: the rest of the editor.
- In the browser console, `__mrfseg.describe()` dumps cameras, planes, datasets and actors (each
  slice actor's sampled plane and where it is drawn): debug alignment with numbers, not by eye.
