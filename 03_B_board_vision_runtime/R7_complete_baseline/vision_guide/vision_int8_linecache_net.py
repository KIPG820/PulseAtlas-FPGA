"""OpenCV adapter for the R7 prepared INT8 graph backend."""

from __future__ import annotations

import ctypes
from pathlib import Path

from vision_int8_graph import Int8Graph
from vision_int8_prepared_pl import LineCacheInt8PlBackend


class LineCacheInt8PoseNet:
    def __init__(self, overlay, graph_dir: str | Path, native_library: str | Path):
        self.backend = LineCacheInt8PlBackend(overlay, batch_mmio=True)
        self.backend.native = ctypes.CDLL(str(native_library))
        self.graph = Int8Graph(graph_dir, self.backend, separable_pool=True)

    def setPreferableBackend(self, _backend):
        return None

    def getUnconnectedOutLayersNames(self):
        return tuple(self.graph.manifest["outputs"])

    def forward_frame(self, frame):
        return self.graph.run(self.graph.make_input(frame))
