# 12-lead ECG model training

This directory implements the first stages of the plan in docs/神经网络训练实施指导书.md: PTB-XL audit/cache, a fixed preprocessing contract, and the FP32 1D-ResNet training baseline. Outputs are research/competition estimates, not clinical diagnoses.

## Environment

Use the existing Conda environment named `zsd` (Python 3.8). The verified training setup is PyTorch 2.4.1+cu121 on the RTX 4060 Laptop GPU. Install the remaining Python packages with `python -m pip install -r nn/requirements.txt`; install the matching PyTorch CUDA build separately with `python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`. Check `torch.cuda.is_available()` before starting. Do not place PTB-XL under version control.

## Data

Download PTB-XL 1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/). The metadata audit saves the record manifest; the parallel downloader fetches only labeled 500 Hz signals in folds 1–9 from PhysioNet's public mirror. Fold 10 stays out of training and validation. `LICENSE.txt` is retained locally; cite the PTB-XL dataset and comply with its CC BY 4.0 terms if you redistribute derived material.

Example:

    conda activate zsd
    python -m pip install -r nn/requirements.txt
    python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    python nn/scripts/prepare_ptbxl.py --dataset-root nn/data/ptb-xl --metadata-only
    python nn/scripts/download_ptbxl_signals.py --folds 1-9 --workers 32
    python nn/scripts/verify_ptbxl_signals.py --folds 1-9
    python nn/scripts/prepare_ptbxl.py --dataset-root nn/data/ptb-xl --skip-download
    python nn/scripts/train.py --config nn/configs/train_ptbxl.yaml
    # Only after freezing the checkpoint and fold-9 thresholds:
    python nn/scripts/download_ptbxl_signals.py --folds 10 --workers 32
    python nn/scripts/prepare_ptbxl.py --dataset-root nn/data/ptb-xl --test-only-cache --skip-download
    python nn/scripts/evaluate.py --config nn/configs/train_ptbxl.yaml
    python nn/scripts/export_model_manifest.py

Preparation audits the patient-level official folds before writing fold-specific NumPy memory maps. Fold 1–8 are training, fold 9 is validation, and fold 10 is held back for one final evaluation. The filtered cache can take several GB. The individual-file downloader can resume by skipping files with verified expected sizes.

## Current scope

- prepare_ptbxl.py: metadata/label/fold audit and optional signal cache.
- preprocess.py: fixed 500 SPS, canonical lead order, and project IP3 reference filter.
- models/resnet1d.py: the agreed five-label 1D ResNet.
- scripts/train.py: FP32 training and fold-9 threshold selection.
- scripts/quantize_ptq.py: frozen-checkpoint FX static INT8 PTQ. It calibrates on folds 1-8, compares against FP32 on fold 9, and exports an x86 TorchScript reference plus hardware-neutral INT8 tensors and a layer quantization contract.

Example quantization run:

    python nn/scripts/quantize_ptq.py --calibration-records 512

This writes `model_ptq_int8_x86.ts`, `model_ptq_int8_weights.npz`, `model_ptq_int8_hw.json`, and `ptq_int8_fold9_metrics.json` under `nn/artifacts`. The x86 TorchScript file is for the CPU PTQ reference only. The NPZ/JSON files expose INT8 weights, int32 biases, scales, zero-points, residual-add settings, and operator geometry for a later bit-accurate HLS/RTL implementation; they are not a bitstream and do not imply FPGA timing/resource results. Fold 10 remains reserved for final evaluation of a frozen deployment candidate.
