"""Strip YOLO26's unsupported ONNX TopK postprocessing for PYNQ OpenCV 4.5.4.

The extracted graph retains the original FP32 neural-network weights and
exports box distances, class logits, mask coefficients, and mask prototypes.
The board-side decoder reproduces the model's box decoding and top-300 selection.

Usage (development host with ``onnx`` installed):
    python export_psvision_cv454_model.py source.onnx destination.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


OUTPUTS = (
    "/model.23/Concat_output_0",      # DFL distances, in feature-grid units
    "/model.23/Concat_1_output_0",    # class logits
    "/model.23/Concat_2_output_0",    # mask coefficients
    "output1",                        # mask prototypes
)
OUTPUT_SHAPES = (
    (1, 4, 6300),
    (1, 80, 6300),
    (1, 32, 6300),
    (1, 32, 120, 160),
)


def export_compatible_model(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must be different files")
    if not source.is_file():
        raise FileNotFoundError("source ONNX model not found: {}".format(source))

    inferred = shape_inference.infer_shapes(onnx.load(str(source), load_external_data=True))
    available = {
        value.name
        for value in list(inferred.graph.value_info)
        + list(inferred.graph.output)
        + list(inferred.graph.input)
    }
    missing = set(OUTPUTS).difference(available)
    if missing:
        raise ValueError("source model does not expose expected YOLO26 tensors: {}".format(
            ", ".join(sorted(missing))
        ))
    input_names = [value.name for value in inferred.graph.input]
    if input_names != ["images"]:
        raise ValueError("expected a single fixed-shape 'images' input, got {}".format(input_names))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.onnx")
    try:
        onnx.utils.extract_model(
            str(source), str(temporary), input_names, list(OUTPUTS), check_model=True
        )
        model = onnx.load(str(temporary), load_external_data=True)
        if len(model.graph.output) != len(OUTPUTS):
            raise RuntimeError("extracted graph has an unexpected output count")
        for value, expected_name, expected_shape in zip(model.graph.output, OUTPUTS, OUTPUT_SHAPES):
            if value.name != expected_name:
                raise RuntimeError("output order changed unexpectedly: {}".format(value.name))
            dims = value.type.tensor_type.shape.dim
            if len(dims) != len(expected_shape):
                raise RuntimeError("unexpected output rank for {}".format(value.name))
            for dim, size in zip(dims, expected_shape):
                dim.ClearField("dim_param")
                dim.dim_value = size

        operators = {node.op_type for node in model.graph.node}
        if "TopK" in operators:
            raise RuntimeError("unsupported TopK operation remains in extracted graph")
        onnx.checker.check_model(model)
        onnx.save(model, str(temporary))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("Created OpenCV-4.5.4-compatible graph: {}".format(destination))
    print("Nodes: {}; opset: {}; outputs: {}".format(
        len(model.graph.node),
        model.opset_import[0].version,
        ", ".join(OUTPUTS),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_compatible_model(args.source, args.destination)


if __name__ == "__main__":
    main()
