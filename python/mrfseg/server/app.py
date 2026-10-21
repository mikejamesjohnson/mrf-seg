"""The backend's HTTP side: routes, request parsing and responses. The API is listed in
`mrfseg.server`'s docstring."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData

from ..errors import InputError, is_number
from ..extractors import FEATURE_EXTRACTORS, describe_extractor
from ..features import FeatureExtractor
from ..grid import axes, voxels
from ..image_io import ImageReadError, read_image
from ..segmentation import MRFSegmentation, MRFSettings
from .encoding import MAPS, little_endian, mesh_bytes
from .sessions import NotReady, Session, Sessions

DEFAULT_PORT = int(os.environ.get("MRFSEG_BACKEND_PORT", "8004"))
#: Each session holds an image, its features and three maps (about 2 GB for a 512 x 512 x 361
#: CT), and a page reload leaves its session behind, so only the most recent few are kept.
MAX_SESSIONS = 2
MAX_BODY_BYTES = 8 << 20
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SETTINGS_FIELDS = {f.name for f in dataclasses.fields(MRFSettings)}
#: Exit status when the port is taken, so the launcher can tell it from a crash.
EXIT_PORT_IN_USE = 2


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    # Strip the port; an IPv6 host is bracketed, so its colons are inside the brackets.
    name = host[: host.index("]") + 1] if host.startswith("[") and "]" in host else host.split(":")[0]
    return name.lower() in LOOPBACK_HOSTS


def number_field(body: dict, key: str, default: float | None) -> float | None:
    """`body[key]` as a finite number, `default` when it is missing, or a 400."""
    value = body.get(key)
    if value is None:
        return default
    if not is_number(value):
        raise HttpError(400, f'"{key}" must be a finite number')
    return float(value)


def parse_settings(body: dict, current: MRFSettings) -> MRFSettings:
    raw = body.get("settings")
    if raw is None:
        return current
    if not isinstance(raw, dict):
        raise HttpError(400, '"settings" must be an object')
    unknown = set(raw) - SETTINGS_FIELDS
    if unknown:
        raise HttpError(400, f"Unknown settings: {', '.join(sorted(unknown))}")
    try:
        return dataclasses.replace(current, **raw)
    except (TypeError, ValueError) as error:  # InputError is a ValueError
        raise HttpError(400, str(error)) from None


def parse_points(body: dict) -> tuple[np.ndarray, np.ndarray]:
    points, labels = body.get("points"), body.get("labels")
    if not (isinstance(points, list) and all(isinstance(p, list) and len(p) == 3 and all(map(is_number, p)) for p in points)):
        raise HttpError(400, '"points" must be a list of [x, y, z] numbers')
    if not (isinstance(labels, list) and all(type(label) is int and label in (-1, 1) for label in labels)):
        raise HttpError(400, '"labels" must be a list of 1 (inside) and -1 (outside)')
    if len(points) != len(labels):
        raise HttpError(400, f"{len(points)} points but {len(labels)} labels")
    return np.array(points, dtype=np.float64).reshape(-1, 3), np.array(labels, dtype=np.int64)


def geometry_of(image: vtkImageData) -> dict:
    """The grid, with the direction listed as the editor does: i axis, j axis, k axis."""
    return {
        "dimensions": list(image.GetDimensions()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": axes(image).T.ravel().tolist(),
    }


class Handler(BaseHTTPRequestHandler):
    server: Server
    server_version = "MRFSeg/0.1"

    # --- routes ----------------------------------------------------------------------------

    def health(self) -> None:
        self.send_json(200, {"ok": True})

    def create_session(self) -> None:
        body = self.read_json()
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise HttpError(400, 'Expected {"path": "/absolute/path/to/image"}')
        if not Path(path).is_absolute():
            raise HttpError(400, f"Not an absolute path: {path}")
        settings = parse_settings(body, MRFSettings())
        try:
            image = read_image(path)
        except ImageReadError as error:
            raise HttpError(400, str(error)) from None
        segmentation = MRFSegmentation(image, settings, self.server.extractor())
        segmentation.check_settings()  # a grid too large for memory is refused here, not in the background
        session = Session(segmentation)
        self.server.sessions.add(session)
        session.start_features()
        self.log(f"session {session.id}: {path}")
        self.send_json(
            201,
            {
                "id": session.id,
                "geometry": geometry_of(image),
                # With what the request left to the image chosen, for the editor to show.
                "settings": dataclasses.asdict(segmentation.settings),
                "features": session.features.report(),
            },
        )

    def extractors(self) -> None:
        """Each feature extractor the settings can name: whether it can run here, the settings
        to start from with it, its native grid (`describe_extractor`)."""
        self.send_json(200, {"extractors": {name: describe_extractor(name) for name in FEATURE_EXTRACTORS}})

    def session_status(self, session_id: str) -> None:
        session = self.session(session_id)
        self.send_json(200, {"id": session.id, "features": session.features.report(), "job": session.job.report()})

    def _checked_request(self, session: Session) -> tuple[np.ndarray, np.ndarray, MRFSettings]:
        """The points, labels and settings of a Generate, checked here, so a request that cannot
        run is a 400 and never touches the last result."""
        body = self.read_json()
        settings = parse_settings(body, session.segmentation.settings)
        points, labels = parse_points(body)
        points, labels = session.segmentation.check_points(points, labels)
        session.segmentation.check_settings(settings)
        return points, labels, settings

    def segment(self, session_id: str) -> None:
        session = self.session(session_id)
        points, labels, settings = self._checked_request(session)
        job = session.start_job(points, labels, settings)
        if job is None:
            raise HttpError(409, "A segmentation is already running")
        self.log(f"session {session.id}: segmenting with {len(labels)} points")
        self.send_json(202, {"job": job.report()})

    def get_map(self, session_id: str, name: str) -> None:
        session = self.session(session_id)
        encoding = parse_qs(urlsplit(self.path).query).get("encoding", ["float32"])[0]
        if encoding not in ("float32", "compact"):
            raise HttpError(400, f"Unknown encoding {encoding!r}: float32 or compact")
        values = session.map_values(name, encoding == "compact")
        if values is None:
            raise NotReady("No segmentation: generate one first, or wait for the one being generated")
        self.send_bytes(little_endian(values))

    def make_surface(self, session_id: str) -> None:
        session = self.session(session_id)
        body = self.read_json()
        source, level = body.get("source", "probability"), number_field(body, "level", None)
        started = time.perf_counter()
        surface = session.build_surface(source, level)
        summary = {"source": source, "level": level, "points": surface.GetNumberOfPoints(), "triangles": surface.GetNumberOfPolys()}
        self.send_json(200, {**summary, "seconds": time.perf_counter() - started})

    def get_surface(self, session_id: str) -> None:
        surface = self.session(session_id).products.get("surface")
        if surface is None:
            raise NotReady("No surface yet: build one first")
        self.send_bytes(mesh_bytes(surface))

    def make_labelmap(self, session_id: str) -> None:
        session = self.session(session_id)
        body = self.read_json()
        threshold = number_field(body, "threshold", 0.5)
        connected = body.get("connected", False)
        if not isinstance(connected, bool):
            raise HttpError(400, '"connected" must be true or false')
        started = time.perf_counter()
        labels = voxels(session.build_labelmap(threshold, connected))
        summary = {"threshold": threshold, "connected": connected, "voxels": int(np.count_nonzero(labels))}
        self.send_json(200, {**summary, "seconds": time.perf_counter() - started})

    def get_labelmap(self, session_id: str) -> None:
        labelmap = self.session(session_id).products.get("labelmap")
        if labelmap is None:
            raise NotReady("No labelmap yet: build one first")
        self.send_bytes(voxels(labelmap).ravel())

    def delete_session(self, session_id: str) -> None:
        if not self.server.sessions.remove(session_id):
            raise HttpError(404, f"No session {session_id}")
        self.send_response(204)
        self.end_headers()

    ROUTES: list[tuple[str, re.Pattern, Callable]] = [
        ("GET", re.compile(r"/health"), health),
        ("GET", re.compile(r"/extractors"), extractors),
        ("POST", re.compile(r"/sessions"), create_session),
        ("GET", re.compile(r"/sessions/(\w+)"), session_status),
        ("POST", re.compile(r"/sessions/(\w+)/segmentation"), segment),
        ("GET", re.compile(rf"/sessions/(\w+)/segmentation/({'|'.join(MAPS)})"), get_map),
        ("POST", re.compile(r"/sessions/(\w+)/surface"), make_surface),
        ("GET", re.compile(r"/sessions/(\w+)/surface"), get_surface),
        ("POST", re.compile(r"/sessions/(\w+)/labelmap"), make_labelmap),
        ("GET", re.compile(r"/sessions/(\w+)/labelmap"), get_labelmap),
        ("DELETE", re.compile(r"/sessions/(\w+)"), delete_session),
    ]

    # --- plumbing --------------------------------------------------------------------------

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def do_DELETE(self) -> None:
        self.dispatch("DELETE")

    def dispatch(self, method: str) -> None:
        try:
            self.check_caller(method)
            path = urlsplit(self.path).path.rstrip("/") or "/"
            matches = [(route_method, route, pattern.fullmatch(path)) for route_method, pattern, route in self.ROUTES]
            for route_method, route, match in matches:
                if match and route_method == method:
                    route(self, *match.groups())
                    return
            if any(match for _, _, match in matches):
                raise HttpError(405, f"{method} is not allowed on {path}")
            raise HttpError(404, f"No such endpoint: {method} {path}")
        except HttpError as error:
            self.send_json(error.status, {"error": str(error)})
        except InputError as error:  # the core's: the request cannot be done as asked
            self.send_json(400, {"error": str(error)})
        except NotReady as error:
            self.send_json(409, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the editor went away mid-response
        except Exception as error:  # a bug: report it rather than drop the connection
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(error).__name__}: {error}"})

    def check_caller(self, method: str) -> None:
        """This computer, and not a web page. The Host check stops DNS rebinding; the editor's
        server calls without an Origin, which every browser request carries across sites; and a
        POST must be JSON, which a page cannot send here without a preflight this never allows."""
        if not is_loopback(self.headers.get("Host")):
            raise HttpError(403, "The segmentation backend only serves this computer")
        if self.headers.get("Origin"):
            raise HttpError(403, "The segmentation backend only serves the editor's server")
        if method == "POST" and not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise HttpError(415, "Requests must be JSON")

    def session(self, session_id: str) -> Session:
        session = self.server.sessions.get(session_id)
        if session is None:
            raise HttpError(404, f"No session {session_id}: it was closed, or evicted for a newer one")
        return session

    def read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise HttpError(400, "Bad Content-Length") from None
        if length < 0:
            raise HttpError(400, "Bad Content-Length")
        if length > MAX_BODY_BYTES:
            raise HttpError(413, "Request body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):  # JSONDecodeError, or a number too long to convert
            raise HttpError(400, "Request body is not JSON") from None
        if not isinstance(body, dict):
            raise HttpError(400, "Request body must be a JSON object")
        return body

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode()  # NaN is not JSON; better to fail here
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data: bytes | np.ndarray) -> None:
        body = memoryview(data).cast("B")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        chunk = 1 << 23
        for start in range(0, len(body), chunk):
            self.wfile.write(body[start : start + chunk])

    def log(self, message: str) -> None:
        if not self.server.quiet:
            print(f"[mrfseg] {message}", file=sys.stderr, flush=True)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- the base class's name
        # A malformed request line leaves `command` and `path` unset.
        command, path = getattr(self, "command", ""), urlsplit(getattr(self, "path", "")).path
        # Polls arrive several times a second; only requests that do something are logged.
        if command == "GET" and re.fullmatch(r"/(health|sessions/\w+)", path):
            return
        self.log(f"{command} {path} {args[1] if len(args) > 1 else ''}")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, port: int, quiet: bool = False, extractor: Callable[[], FeatureExtractor | None] | None = None
    ) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.sessions = Sessions(MAX_SESSIONS)
        self.quiet = quiet
        #: Makes each session's feature extractor; None means the one its settings choose. Tests pass a stub.
        self.extractor = extractor or (lambda: None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mrfseg.server", description="The MRFSeg segmentation backend.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    port = parser.parse_args().port
    try:
        server = Server(port)
    except OSError as error:
        print(f"[mrfseg] cannot listen on 127.0.0.1:{port} ({error.strerror}). Is another backend running?", file=sys.stderr)
        sys.exit(EXIT_PORT_IN_USE)
    print(f"[mrfseg] segmentation backend on http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
