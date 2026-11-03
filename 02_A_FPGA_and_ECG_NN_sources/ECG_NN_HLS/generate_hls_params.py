"""Generate deterministic HLS weight tables from the validated PTQ contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LAYERS = [
    "stem.0",
    "stage1.0.conv1", "stage1.0.conv2",
    "stage1.1.conv1", "stage1.1.conv2",
    "stage2.0.projection.0", "stage2.0.conv1", "stage2.0.conv2",
    "stage2.1.conv1", "stage2.1.conv2",
    "stage3.0.projection.0", "stage3.0.conv1", "stage3.0.conv2",
    "stage3.1.conv1", "stage3.1.conv2",
    "classifier",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_lines(values, per_line: int = 20) -> str:
    numbers = [str(int(value)) for value in values]
    return "\n".join(
        "    " + ", ".join(numbers[index:index + per_line]) + ","
        for index in range(0, len(numbers), per_line)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=ROOT / "nn" / "artifacts" / "model_ptq_int8_hw.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "nn" / "pl" / "hls" / "ecg_nn_params.h")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    weight_path = args.contract.parent / contract["weights_file"]
    arrays = np.load(weight_path)
    layers = contract["quantized_layers"]
    if [layer["module"] for layer in layers] != EXPECTED_LAYERS:
        raise ValueError("PTQ layer order does not match the verified 1D ResNet")

    graph_shape = {}
    graph_quant = {}
    graph_shape["x"] = (12, 5000)
    weight_values, bias_values, multiplier_values = [], [], []
    descriptors = []
    add_descriptors = []
    layer_by_node = {layer["node"]: layer for layer in layers}

    for op in contract["graph_ops"]:
        kind, node = op["op"], op["node"]
        if kind == "input_affine_quantize":
            graph_shape[node] = (12, 5000)
            graph_quant[node] = (float(op["input_scale"]), int(op["input_zero_point"]))
        elif kind == "quantized_layer":
            layer = layer_by_node[node]
            source = layer["input_node"]
            input_shape = graph_shape[source]
            key = layer["module"].replace(".", "_")
            weight = arrays[key + "_weight_int8"]
            bias = arrays[key + "_bias_int32"]
            if np.any(arrays[key + "_weight_zero_points_i32"] != 0):
                raise ValueError("nonzero weight zero-point unsupported in HLS prototype")
            ratios = np.asarray(layer["requantization_multiplier"], dtype=np.float64)
            if ratios.size != weight.shape[0]:
                raise ValueError("per-output-channel multiplier count mismatch")
            q31 = np.rint(ratios * (1 << 31)).astype(np.int64)
            if np.any(q31 < 0) or np.any(q31 > np.iinfo(np.int32).max):
                raise ValueError("requantization multiplier exceeds Q31 range")
            if len(weight.shape) == 3:
                out_channels, in_channels, kernel = map(int, weight.shape)
                if input_shape[0] != in_channels:
                    raise ValueError("input channel mismatch for {}".format(node))
                setting = layer["activation"]
                stride = int(setting["stride"][0])
                pad = int(setting["padding"][0])
                input_length = int(input_shape[1])
                output_length = (input_length + 2 * pad - kernel) // stride + 1
                graph_shape[node] = (out_channels, output_length)
            else:
                out_channels, in_channels = map(int, weight.shape)
                if input_shape != (in_channels,):
                    raise ValueError("linear input shape mismatch")
                kernel, stride, pad = 1, 1, 0
                input_length, output_length = 1, 1
                graph_shape[node] = (out_channels,)

            descriptors.append({
                "name": layer["module"],
                "in_channels": in_channels,
                "out_channels": out_channels,
                "input_length": input_length,
                "output_length": output_length,
                "kernel": kernel,
                "stride": stride,
                "padding": pad,
                "weight_offset": len(weight_values),
                "parameter_offset": len(bias_values),
                "input_zero_point": int(layer["input_zero_point"]),
                "output_zero_point": int(layer["output_zero_point"]),
                "fused_relu": int("ConvReLU" in layer["operator"]),
            })
            weight_values.extend(weight.reshape(-1).astype(int).tolist())
            bias_values.extend(bias.astype(int).tolist())
            multiplier_values.extend(q31.astype(int).tolist())
            graph_quant[node] = (float(layer["output_scale"]),
                                 int(layer["output_zero_point"]))
        elif kind == "int8_add_then_relu":
            a, b = op["input_nodes"]
            if graph_shape[a] != graph_shape[b]:
                raise ValueError("residual branches have different shapes")
            scale = float(op["output_scale"])
            am = int(np.rint(graph_quant[a][0] / scale * (1 << 31)))
            bm = int(np.rint(graph_quant[b][0] / scale * (1 << 31)))
            add_descriptors.append({
                "node": node,
                "channels": int(graph_shape[a][0]),
                "length": int(graph_shape[a][1]),
                "a_multiplier_q31": am,
                "b_multiplier_q31": bm,
                "a_zero_point": int(graph_quant[a][1]),
                "b_zero_point": int(graph_quant[b][1]),
                "output_zero_point": int(op["output_zero_point"]),
            })
            graph_shape[node] = graph_shape[a]
            graph_quant[node] = (scale, int(op["output_zero_point"]))
        elif kind in {"identity", "dropout_identity_in_eval"}:
            graph_shape[node] = graph_shape[op["input_node"]]
            graph_quant[node] = graph_quant[op["input_node"]]
        elif kind == "adaptive_avg_pool1d":
            source = op["input_node"]
            graph_shape[node] = (graph_shape[source][0], 1)
            graph_quant[node] = graph_quant[source]
        elif kind == "squeeze":
            source = op["input_node"]
            graph_shape[node] = (graph_shape[source][0],)
            graph_quant[node] = graph_quant[source]
        elif kind == "dequantize_output":
            source = op["input_node"]
            graph_shape[node] = graph_shape[source]
            graph_quant[node] = graph_quant[source]
        elif kind == "model_output":
            pass
        else:
            raise ValueError("unsupported graph operation {}".format(kind))

    if len(descriptors) != 16 or len(add_descriptors) != 6:
        raise ValueError("expected exactly 16 weighted layers and six residual adds")
    if max(d["out_channels"] * d["output_length"] for d in descriptors[:-1]) > 20032:
        raise ValueError("activation buffer capacity is insufficient")
    if len(bias_values) != len(multiplier_values):
        raise ValueError("bias and Q31 multiplier arrays differ in length")

    layer_rows = "\n".join(
        "    {{{in_channels}, {out_channels}, {input_length}, {output_length}, "
        "{kernel}, {stride}, {padding}, {weight_offset}, {parameter_offset}, "
        "{input_zero_point}, {output_zero_point}, {fused_relu}}}, // {name}".format(**d)
        for d in descriptors
    )
    add_rows = "\n".join(
        "    {{{channels}, {length}, {a_multiplier_q31}LL, {b_multiplier_q31}LL, "
        "{a_zero_point}, {b_zero_point}, {output_zero_point}}}, // {node}".format(**d)
        for d in add_descriptors
    )
    packed_weights = [
        sum((int(v) & 255) << (8 * lane)
            for lane, v in enumerate(weight_values[start:start + 4]))
        for start in range(0, len(weight_values), 4)
    ]
    weight_chunks = [packed_weights[:16384],
                     packed_weights[16384:32768],
                     packed_weights[32768:]]
    weight_tables = "\n".join(
        "static const uint32_t kPackedWeights{}[{}] = {{\n{}\n}};".format(
            i, len(chunk), array_lines(chunk))
        for i, chunk in enumerate(weight_chunks)
    )
    content = f"""// Generated by generate_hls_params.py. Do not edit by hand.
// PTQ contract SHA256: {sha256_file(args.contract)}
// Weight archive SHA256: {sha256_file(weight_path)}
#ifndef ECG_NN_PARAMS_H
#define ECG_NN_PARAMS_H
#include <stdint.h>

struct LayerDesc {{
    int in_channels, out_channels, input_length, output_length;
    int kernel, stride, padding, weight_offset, parameter_offset;
    int input_zero_point, output_zero_point, fused_relu;
}};
struct AddDesc {{
    int channels, length;
    int64_t a_multiplier_q31, b_multiplier_q31;
    int a_zero_point, b_zero_point, output_zero_point;
}};

static const int kLayerCount = {len(descriptors)};
static const int kAddCount = {len(add_descriptors)};
static const int kWeightCount = {len(weight_values)};
static const int kParameterCount = {len(bias_values)};
static const int kInputBytes = 60000;
static const int kActivationCapacity = 20032;
static const float kOutputScale = {float(layers[-1]['output_scale']):.17g}f;
static const int kOutputZeroPoint = {int(layers[-1]['output_zero_point'])};

static const LayerDesc kLayers[{len(descriptors)}] = {{
{layer_rows}
}};
static const AddDesc kAdds[{len(add_descriptors)}] = {{
{add_rows}
}};
{weight_tables}
static const int32_t kBiases[{len(bias_values)}] = {{
{array_lines(bias_values)}
}};
static const int32_t kMultipliersQ31[{len(multiplier_values)}] = {{
{array_lines(multiplier_values)}
}};
#endif
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    manifest = {
        "generator": "generate_hls_params.py",
        "source_contract_sha256": sha256_file(args.contract),
        "source_weights_sha256": sha256_file(weight_path),
        "header_sha256": sha256_file(args.output),
        "header_file": args.output.name,
        "layers": descriptors,
        "residual_adds": add_descriptors,
        "weight_count": len(weight_values),
        "parameter_count": len(bias_values),
        "output_scale": float(layers[-1]["output_scale"]),
        "output_zero_point": int(layers[-1]["output_zero_point"]),
    }
    manifest_path = args.output.parent / "ecg_nn_params_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "header": str(args.output),
        "manifest": str(manifest_path),
        "weights": len(weight_values),
        "channels_of_bias_and_scale": len(bias_values),
        "residual_adds": len(add_descriptors),
    }, indent=2))


if __name__ == "__main__":
    main()
