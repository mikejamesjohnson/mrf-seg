"""The segmentation backend, over real HTTP on an ephemeral port, with the stub features."""

import http.client
import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPolyData

from mrfseg import MRFSegmentation, MRFSettings, read_image
from mrfseg.grid import voxels
from mrfseg.server import Server
from mrfseg.server.app import MAX_SESSIONS
from mrfseg.server.encoding import mesh_bytes
from tests.helpers import sphere_points, sphere_volume, stub_extractor, write_metaimage

# As `tests.test_segmentation.FAST`: the contrast affinities on the test sphere, the outside points as seeds.
SETTINGS = {"feature_spacing": 2.0, "device": "cpu"}


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server(port=0, quiet=True, extractor=lambda: stub_extractor)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.image_path = Path(cls.tmp.name) / "sphere.mha"
        # `sphere_volume`'s sphere, moved off the origin, as a file.
        volume, _ = sphere_volume(origin=(-10.0, 4.0, 30.0))
        write_metaimage(
            cls.image_path,
            voxels(volume),
            "MET_SHORT",
            compressed=True,
            dims=volume.GetDimensions(),
            spacing=volume.GetSpacing(),
            origin=volume.GetOrigin(),
            axes=np.eye(3),
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def call(self, method, path, body=None, host=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        sent = {"Host": host or f"127.0.0.1:{self.port}"}
        if body is not None:
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        payload = body if isinstance(body, (bytes, str)) else None if body is None else json.dumps(body)
        connection.request(method, path, body=payload, headers=sent)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), data

    def open_session(self):
        status, _, data = self.call("POST", "/sessions", {"path": str(self.image_path), "settings": SETTINGS})
        self.assertEqual(status, 201, data)
        return json.loads(data)

    def wait(self, session, key):
        """Poll the session until `key` ("features" or "job") is no longer running."""
        for _ in range(600):
            status = json.loads(self.call("GET", f"/sessions/{session}")[2])
            if status[key]["state"] != "running":
                return status[key]
            time.sleep(0.05)
        self.fail(f"{key} still running")

    def points(self):
        """`sphere_points`, which are relative to the sphere, moved into this image's frame."""
        image = read_image(self.image_path)
        centre = np.array(image.GetOrigin()) + (np.array(image.GetDimensions()) - 1) * np.array(image.GetSpacing()) / 2
        points, labels = sphere_points()
        return (points + centre).tolist(), labels.tolist()

    def segment(self, session, points=None, labels=None, settings=SETTINGS):
        if points is None:
            points, labels = self.points()
        status, _, data = self.call(
            "POST", f"/sessions/{session}/segmentation", {"points": points, "labels": labels, "settings": settings}
        )
        return status, json.loads(data)

    def test_health(self):
        self.assertEqual(self.call("GET", "/health")[0], 200)

    def test_session_reads_the_image_and_extracts_its_features(self):
        opened = self.open_session()
        self.assertEqual(opened["geometry"]["dimensions"], [24, 22, 18])
        self.assertEqual((opened["settings"]["feature_spacing"], opened["settings"]["contrast"]), (2.0, 40.0))
        features = self.wait(opened["id"], "features")
        self.assertEqual(features["state"], "done", features)
        self.assertEqual((features["result"]["channels"], features["result"]["spacing"], features["result"]["extractor"]), (3, 2.0, "basic"))
        self.assertEqual(features["result"]["dims"], [24, 22, 22])  # 17 x 2.5 mm is 21.25 steps of 2 mm

    def test_a_session_left_to_the_image_says_what_it_chose(self):
        # The test image's voxels are 2 x 2 x 2.5 mm.
        status, _, data = self.call("POST", "/sessions", {"path": str(self.image_path), "settings": {"device": "cpu", "feature_spacing": None}})
        self.assertEqual(status, 201, data)
        settings = json.loads(data)["settings"]
        self.assertEqual((settings["feature_spacing"], settings["max_iterations"], settings["background_seeds"]), (2.0, 1000, True))

    def test_generate_then_fetch_the_two_maps(self):
        session = self.open_session()["id"]
        self.assertEqual(self.call("GET", f"/sessions/{session}/segmentation/probability")[0], 409)
        status, body = self.segment(session)
        self.assertEqual(status, 202, body)
        job = self.wait(session, "job")
        self.assertEqual(job["state"], "done", job)
        result = job["result"]
        self.assertEqual(result["points"], 10)
        self.assertEqual(set(result["maps"]), {"affinity", "probability"})
        self.assertEqual(set(result["inference"]), {"iterations", "residual", "converged", "components", "seeds"})
        self.assertEqual(result["inference"]["seeds"], {"inside": 4, "outside": 6, "background": 0})
        self.assertEqual(set(result["seconds"]), {"features", "affinities", "seeds", "solve", "maps"})

        points, labels = self.points()
        expected = MRFSegmentation(read_image(self.image_path), MRFSettings(**SETTINGS), stub_extractor).generate_segmentation(
            np.array(points), np.array(labels)
        )
        for name, dtype in (("affinity", "<i2"), ("probability", "u1")):
            with self.subTest(name):
                status, content_type, data = self.call("GET", f"/sessions/{session}/segmentation/{name}")
                self.assertEqual((status, content_type), (200, "application/octet-stream"))
                values = np.frombuffer(data, dtype="<f4")
                want = voxels(getattr(expected, name)).ravel()
                np.testing.assert_allclose(values, want, atol=1e-4)  # the same inference, run twice

                status, _, data = self.call("GET", f"/sessions/{session}/segmentation/{name}?encoding=compact")
                scale = result["maps"][name]["scale"]
                np.testing.assert_allclose(np.frombuffer(data, dtype=dtype) * scale, values, atol=scale / 2 + 1e-6)
        self.assertEqual(self.call("GET", f"/sessions/{session}/segmentation/probability?encoding=png")[0], 400)

    def test_extractors(self):
        status, _, data = self.call("GET", "/extractors")
        self.assertEqual(status, 200, data)
        described = json.loads(data)["extractors"]
        self.assertEqual(set(described), {"basic", "learned"})
        for name, info in described.items():
            with self.subTest(name):
                self.assertEqual(set(info), {"available", "settings", "native_spacing", "note"})
        self.assertEqual((described["basic"]["available"], described["basic"]["settings"]), (True, None))
        learned = described["learned"]
        if learned["available"]:  # a checkpoint on this machine: the settings to start from, by their names
            self.assertEqual(set(learned["settings"]), {"affinity_scale"})
            self.assertEqual(learned["native_spacing"], 0.5)
        else:
            self.assertIn("checkpoint", learned["note"])

    def test_a_request_that_cannot_run_is_refused_before_it_starts(self):
        session = self.open_session()["id"]
        points, labels = self.points()
        status, body = self.segment(session, points[4:], labels[4:])  # outside points only
        self.assertEqual((status, "inside" in body["error"]), (400, True), body)
        status, body = self.segment(session, settings={**SETTINGS, "feature_spacing": 0.001})  # a grid too large for memory
        self.assertEqual((status, "use at least" in body["error"]), (400, True), body)
        self.assertEqual(self.call("GET", f"/sessions/{session}")[0], 200)  # and the session is as it was

    def test_a_generate_that_fails_keeps_the_last_result_and_says_what_the_features_are(self):
        session = self.open_session()["id"]
        self.segment(session)
        self.assertEqual(self.wait(session, "job")["state"], "done")
        self.assertEqual(self.call("POST", f"/sessions/{session}/labelmap", {"threshold": 0.5})[0], 200)
        # New features are needed, and extracting them fails.
        stub_extractor.fail_with = RuntimeError("no features today")
        self.segment(session, settings={**SETTINGS, "feature_spacing": 3.0})
        job = self.wait(session, "job")
        self.assertEqual((job["state"], "no features today" in job["message"]), ("error", True), job)
        features = json.loads(self.call("GET", f"/sessions/{session}")[2])["features"]
        self.assertEqual((features["state"], "no features today" in features["message"]), ("error", True), features)
        # The last result, and what was built from it, are still there.
        self.assertEqual(self.call("GET", f"/sessions/{session}/segmentation/probability")[0], 200)
        self.assertEqual(self.call("GET", f"/sessions/{session}/labelmap")[0], 200)
        self.assertEqual(self.call("POST", f"/sessions/{session}/surface", {"source": "probability", "level": 0.3})[0], 200)
        # Once a Generate succeeds, the features report follows the new grid.
        self.segment(session, settings={**SETTINGS, "feature_spacing": 3.0})
        self.assertEqual(self.wait(session, "job")["state"], "done")
        features = json.loads(self.call("GET", f"/sessions/{session}")[2])["features"]
        self.assertEqual((features["state"], features["result"]["spacing"]), ("done", 3.0), features)

    def test_surface(self):
        session = self.open_session()["id"]
        self.assertEqual(self.call("POST", f"/sessions/{session}/surface", {"source": "probability"})[0], 409)
        self.segment(session)
        self.wait(session, "job")
        status, _, data = self.call("POST", f"/sessions/{session}/surface", {})
        self.assertEqual(status, 200, data)
        summary = json.loads(data)
        status, _, mesh = self.call("GET", f"/sessions/{session}/surface")
        self.assertEqual(status, 200)
        points, triangles = struct.unpack("<II", mesh[:8])
        self.assertEqual((points, triangles), (summary["points"], summary["triangles"]))
        self.assertEqual(len(mesh), 8 + points * 12 * 2 + triangles * 12)
        indices = np.frombuffer(mesh[8 + points * 24 :], dtype="<u4")
        self.assertLess(int(indices.max()), points)
        self.assertEqual(self.call("POST", f"/sessions/{session}/surface", {"source": "probability", "level": 1.5})[0], 400)
        self.assertEqual(self.call("POST", f"/sessions/{session}/surface", {"source": "latent"})[0], 400)

    def test_labelmap(self):
        session = self.open_session()["id"]
        self.assertEqual(self.call("POST", f"/sessions/{session}/labelmap", {"threshold": 0.5})[0], 409)
        self.assertEqual(self.call("GET", f"/sessions/{session}/labelmap")[0], 409)
        self.segment(session)
        self.wait(session, "job")
        status, _, data = self.call("POST", f"/sessions/{session}/labelmap", {"threshold": 0.5})
        self.assertEqual(status, 200, data)
        summary = json.loads(data)
        status, content_type, labels = self.call("GET", f"/sessions/{session}/labelmap")
        self.assertEqual((status, content_type), (200, "application/octet-stream"))
        labels = np.frombuffer(labels, dtype=np.uint8)
        self.assertEqual(labels.size, 24 * 22 * 18)
        self.assertEqual(set(np.unique(labels)), {0, 1})
        self.assertEqual(int(labels.sum()), summary["voxels"])
        for bad in ({"threshold": 1.5}, {"threshold": "half"}, {"threshold": True}):
            with self.subTest(bad):
                self.assertEqual(self.call("POST", f"/sessions/{session}/labelmap", bad)[0], 400)

    def test_an_empty_surface_is_a_header_of_zeros(self):
        self.assertEqual(mesh_bytes(vtkPolyData()), struct.pack("<II", 0, 0))

    def test_bad_requests(self):
        session = self.open_session()["id"]
        cases = [
            ("POST", "/sessions", {"path": "relative/ct.mha"}, 400),
            ("POST", "/sessions", {"path": str(self.image_path.with_name("missing.mha"))}, 400),
            ("POST", "/sessions", {"path": str(self.image_path), "settings": {"colour": "blue"}}, 400),
            ("POST", "/sessions", {"path": str(self.image_path), "settings": {"max_iterations": -1}}, 400),
            ("POST", f"/sessions/{session}/segmentation", {"points": [[0, 0]], "labels": [1]}, 400),
            ("POST", f"/sessions/{session}/segmentation", {"points": [[0, 0, 0]], "labels": [1, -1]}, 400),
            ("POST", "/sessions/nope/segmentation", {"points": [], "labels": []}, 404),
            ("GET", "/sessions/nope", None, 404),
            ("GET", "/elsewhere", None, 404),
            ("GET", "/sessions", None, 405),
        ]
        for method, path, body, want in cases:
            with self.subTest(f"{method} {path} {body}"):
                status, _, data = self.call(method, path, body)
                self.assertEqual(status, want, data)
                self.assertIn("error", json.loads(data))

    def test_only_this_computer(self):
        self.assertEqual(self.call("GET", "/health", host="evil.example:8004")[0], 403)

    def test_not_from_a_web_page(self):
        # A page can POST text/plain without a preflight, and its browser adds an Origin.
        page = {"Content-Type": "text/plain"}
        self.assertEqual(self.call("POST", "/sessions", json.dumps({"path": str(self.image_path)}), headers=page)[0], 415)
        self.assertEqual(self.call("GET", "/health", headers={"Origin": "https://evil.example"})[0], 403)

    def test_what_json_can_smuggle_in(self):
        session = self.open_session()["id"]
        points, labels = self.points()
        cases = {
            "fractional label": (points, [1.9] + labels[1:]),
            "label as a string": (points, ["-1"] + labels[1:]),
            "label as a bool": (points, [True] + labels[1:]),
            "coordinate as a string": ([["1", 0, 0]] + points[1:], labels),
            "coordinate not finite": ([[1e400, 0, 0]] + points[1:], labels),
        }
        for what, (bad_points, bad_labels) in cases.items():
            with self.subTest(what):
                status, data = self.segment(session, bad_points, bad_labels)
                self.assertEqual(status, 400, data)
        for settings in ({"background_seeds": "false"}, {"max_iterations": True}, {"contrast": 1e400}, {"max_iterations": 10**400}, []):
            with self.subTest(settings=settings):
                self.assertEqual(self.segment(session, points, labels, settings=settings)[0], 400)

    def test_bodies_and_headers(self):
        self.assertEqual(self.call("POST", "/sessions", "{}", headers={"Content-Length": "abc"})[0], 400)
        # Refused on its header, before any of it is read.
        self.assertEqual(self.call("POST", "/sessions", "{}", headers={"Content-Length": str(9 << 20)})[0], 413)

    def slow_job(self, session):
        """A Generate long enough to still be running for the next few requests: its features
        are new (another grid), and the stub takes its time over them."""
        stub_extractor.slow_for = 3.0
        status, body = self.segment(session, settings={**SETTINGS, "feature_spacing": 2.5})
        self.assertEqual(status, 202, body)

    def test_while_generating(self):
        session = self.open_session()["id"]
        self.wait(session, "features")
        self.slow_job(session)
        started = time.perf_counter()
        self.assertEqual(self.call("GET", f"/sessions/{session}/segmentation/probability")[0], 409)
        self.assertLess(time.perf_counter() - started, 2.0)  # an answer, not a wait for the job
        self.assertEqual(self.segment(session)[0], 409)  # one Generate at a time
        self.assertEqual(self.call("POST", f"/sessions/{session}/labelmap", {})[0], 409)
        self.assertEqual(self.call("DELETE", f"/sessions/{session}")[0], 204)

    def test_closing_a_session_stops_its_work(self):
        session = self.open_session()["id"]
        self.wait(session, "features")
        self.slow_job(session)
        self.assertEqual(self.call("DELETE", f"/sessions/{session}")[0], 204)
        for _ in range(100):
            if not any(thread.name == f"segment-{session}" for thread in threading.enumerate()):
                break
            time.sleep(0.05)
        else:
            self.fail("the job kept running after its session was closed")

    def test_delete_and_eviction(self):
        session = self.open_session()["id"]
        self.assertEqual(self.call("DELETE", f"/sessions/{session}")[0], 204)
        self.assertEqual(self.call("DELETE", f"/sessions/{session}")[0], 404)

        first = self.open_session()["id"]
        for _ in range(MAX_SESSIONS):
            self.open_session()
        self.assertEqual(self.call("GET", f"/sessions/{first}")[0], 404)


if __name__ == "__main__":
    unittest.main()
