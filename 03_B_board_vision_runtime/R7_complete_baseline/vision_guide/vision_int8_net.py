"""OpenCV-compatible adapter for the full PS/PL integer pose graph."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

from vision_int8_graph import Int8Graph
from vision_int8_pl import Int8PlConvBackend


class Int8PoseNet:
    def __init__(self, overlay, graph_dir: str | Path, native_library: str | Path):
        self.backend = Int8PlConvBackend(overlay, native_library)
        self.graph = Int8Graph(graph_dir, self.backend)

    def setPreferableBackend(self, _backend):
        return None

    def getUnconnectedOutLayersNames(self):
        return tuple(self.graph.manifest["outputs"])

    def forward_frame(self, frame):
        if os.environ.get("PULSEATLAS_VISION_PROFILE") != "1":
            return self.graph.run(self.graph.make_input(frame))
        started = time.perf_counter()
        before = dict(self.backend.totals)
        qinput = self.graph.make_input(frame)
        prepared = time.perf_counter()
        profile = {}
        outputs = self.graph.run(qinput, profile=profile)
        finished = time.perf_counter()
        backend = {key: round(value - before[key], 6)
                   for key, value in self.backend.totals.items()}
        print("VISION_INT8_PROFILE " + json.dumps({
            "input_s": round(prepared - started, 6),
            "graph_s": round(finished - prepared, 6),
            "ops_s": {key: round(value, 6) for key, value in profile.items()},
            "pl_s": backend,
        }, separators=(",", ":")), flush=True)
        return outputs
