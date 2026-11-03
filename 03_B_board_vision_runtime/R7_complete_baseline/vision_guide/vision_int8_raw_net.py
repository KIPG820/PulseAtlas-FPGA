"""OpenCV-compatible adapter for the full UINT8 graph + raw-NCHW PL core."""

from __future__ import annotations

from pathlib import Path

from vision_int8_graph import Int8Graph
from vision_int8_raw_pl import RawNchwInt8PlBackend


class RawNchwInt8PoseNet:
    def __init__(self, overlay, graph_dir: str | Path, native_library: str | Path):
        self.backend = RawNchwInt8PlBackend(overlay, native_library)
        self.graph = Int8Graph(graph_dir, self.backend)

    def setPreferableBackend(self, _backend):
        return None

    def getUnconnectedOutLayersNames(self):
        return tuple(self.graph.manifest["outputs"])

    def forward_frame(self, frame):
        return self.graph.run(self.graph.make_input(frame))


__all__ = ["RawNchwInt8PoseNet"]
