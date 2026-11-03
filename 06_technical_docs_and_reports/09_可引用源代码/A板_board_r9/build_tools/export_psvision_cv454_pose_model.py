"""Extract YOLO11n-pose raw heads for OpenCV 4.5.4 on the PYNQ PS.

Usage after exporting a fixed ONNX model whose height and width are multiples of 32:
    python export_psvision_cv454_pose_model.py source.onnx destination.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


OUTPUTS = (
    "/model.23/Concat_output_0",      # DFL box logits
    "/model.23/Concat_1_output_0",    # person logits
    "/model.23/Concat_2_output_0",    # COCO17 keypoint values
)


def export_compatible_model(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("source and destination must be different files")
    if not source.is_file():
        raise FileNotFoundError("source ONNX model not found: {}".format(source))
    model = shape_inference.infer_shapes(onnx.load(str(source), load_external_data=True))
    if [value.name for value in model.graph.input] != ["images"]:
        raise ValueError("expected a single fixed-shape 'images' input")
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape) != 4 or input_shape[:2] != [1, 3]:
        raise ValueError("expected input shape [1, 3, H, W], got {}".format(input_shape))
    input_height, input_width = input_shape[2:]
    if input_height < 32 or input_width < 32 or input_height % 32 or input_width % 32:
        raise ValueError("input dimensions must be positive multiples of 32, got {}x{}".format(
            input_width, input_height))
    candidate_count = sum(
        (input_width // step) * (input_height // step) for step in (8, 16, 32)
    )
    output_shapes = (
        (1, 64, candidate_count),
        (1, 1, candidate_count),
        (1, 51, candidate_count),
    )
    available = {
        value.name for value in list(model.graph.value_info) + list(model.graph.output)
    }
    missing = set(OUTPUTS).difference(available)
    if missing:
        raise ValueError("source model is missing raw pose heads: {}".format(
            ", ".join(sorted(missing))))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.onnx")
    try:
        onnx.utils.extract_model(
            str(source), str(temporary), ["images"], list(OUTPUTS), check_model=True
        )
        extracted = onnx.load(str(temporary), load_external_data=True)
        if len(extracted.graph.output) != len(OUTPUTS):
            raise RuntimeError("extracted graph has an unexpected output count")
        for value, expected_name, expected_shape in zip(
                extracted.graph.output, OUTPUTS, output_shapes):
            if value.name != expected_name:
                raise RuntimeError("output order changed unexpectedly: {}".format(value.name))
            dims = value.type.tensor_type.shape.dim
            if len(dims) != len(expected_shape):
                raise RuntimeError("unexpected output rank for {}".format(value.name))
            for dim, size in zip(dims, expected_shape):
                dim.ClearField("dim_param")
                dim.dim_value = size
        onnx.checker.check_model(extracted)
        onnx.save(extracted, str(temporary))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("Created OpenCV-4.5.4-compatible pose graph: {}".format(destination))
    print("Input: {}x{}; raw candidates: {}".format(input_width, input_height,
                                                    candidate_count))
    print("Nodes: {}; opset: {}; outputs: {}".format(
        len(extracted.graph.node), extracted.opset_import[0].version, ", ".join(OUTPUTS)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_compatible_model(args.source, args.destination)


if __name__ == "__main__":
    main()
