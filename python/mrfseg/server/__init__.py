"""The segmentation backend: a local HTTP server keeping one live MRFSegmentation per image the
editor opens. `npm run dev` starts it; on its own, from `python/`:

    .venv/bin/python -m mrfseg.server [--port 8004]

The editor reaches it through its `/api/mrf/...` route, so the browser never talks to this
port. JSON in and out, except the maps and the mesh:

    GET    /health                             {"ok": true}
    GET    /extractors                         {"extractors": {name: {"available", "settings", "native_spacing", "note"}}}
                                               (`extractors.describe_extractor`: what to offer, and the settings to start from)
    POST   /sessions                           {"path", "settings"?} -> 201 {"id", "geometry", "settings", "features"}
                                               ("settings" with what the request left to the image chosen)
    GET    /sessions/<id>                      {"id", "features": task, "job": task}, for polling
    POST   /sessions/<id>/segmentation         {"points": [[x, y, z], ...], "labels": [1 | -1, ...], "settings"?} -> 202,
                                               or 400 if the points or settings cannot be used; the job's result is
                                               {"points", "seconds", "inference": {"sweeps", "delta", "converged", "components"}, "maps"}
    GET    /sessions/<id>/segmentation/<map>   unary | coupling | probability, float32, x fastest;
                                               ?encoding=compact for the editor's encoding (`encoding.py`)
    POST   /sessions/<id>/surface              {"source": "probability" | "unary", "level"?} -> mesh summary
    GET    /sessions/<id>/surface              the mesh, binary (`encoding.mesh_bytes`)
    POST   /sessions/<id>/labelmap             {"threshold"?} -> {"threshold", "voxels", "seconds"}
    GET    /sessions/<id>/labelmap             the labelmap, uint8 (1 where p >= threshold), x fastest
    DELETE /sessions/<id>                      204

Feature extraction (started when a session opens) and segmentation run in the background: the
request answers at once, and the status endpoint reports progress and, when done, the result;
a second segmentation while one runs is a 409. A request the session cannot serve yet (no
segmentation, or one being generated) is a 409. The last segmentation and the products built
from it stay until a Generate replaces them, so one that fails leaves them as they were.
Closing a session (DELETE, or eviction for a newer one) stops its work at the next step.

It serves the editor's server only (`app.Handler.check_caller`): it binds to 127.0.0.1, refuses
a Host that is not a loopback name, any request with an Origin, and a POST that is not JSON.
"""

from .app import Server, main

__all__ = ["Server", "main"]
