# Phase 2 — Task 12 (`P2-12`): Vision & Object Detection Model Quantization (ONNX INT8)

## Overview
This module implements an end-to-end **Post-Training Static Quantization (PTQ)** pipeline using **ONNX Runtime** to optimize real-time vision and object detection models (**`RT-DETRv2 R18`**, **`D-FINE small`**, and **`MobileNetV3-Large`**) for edge and mobile deployment.

Large model artifacts and calibration datasets are version-controlled using **DVC (Data Version Control)** backed by **Google Drive**, keeping the Git repository lightweight while maintaining full reproducibility.

The primary objective is to reduce model storage footprint and accelerate CPU/edge inference while strictly keeping detection accuracy degradation within the target threshold (**Estimated mAP Drop $\le 2.00\%$**).

---

## Quantization Strategy & Architecture Insights
Transformer-based object detectors contain highly sensitive attention projections, bounding-box regression heads, and depthwise convolutions that suffer severe accuracy loss under naive full-model INT8 quantization. To overcome this, we implemented a **Layer-Selective Mixed-Precision Static INT8** strategy (`QOperator` format with per-channel weight quantization and `MinMax` activation calibration on 200 representative `640x640` images):

1. **`RT-DETRv2 R18` (ResNet-18 Backbone):**
   - Quantized the 13 deep standard convolutions across the ResNet-18 backbone (`stages.1` through `stages.3`) while keeping the early `embedder`/`stage-0` stem, Hybrid Encoder, and Transformer Decoder heads in `FP32`.
   - Achieved a **40.5% size reduction** (`77.28 MB` $\rightarrow$ `46.00 MB`) and **1.08x CPU speedup** with only **0.56% estimated mAP drop**.

2. **`D-FINE small` (HGNetv2 Backbone + FDR):**
   - Because `D-FINE small` utilizes compact `HGNetv2` blocks (with sensitive depthwise convolutions) and Fine-Grained Distribution Refinement (`FDR`), full backbone quantization degrades localization accuracy.
   - By selectively quantizing the non-terminal standard convolutions in `stages.3` (`stages.3[:-1]`) alongside the `1x1` pointwise convolutions in `stages.2` (14 Conv nodes total), we balanced compute acceleration against `Quantize/Dequantize` overhead—achieving a **1.01x CPU speedup**, **8.6% size reduction** (`39.58 MB` $\rightarrow$ `36.19 MB`), and just **1.00% estimated mAP drop**.

3. **`MobileNetV3-Large` (Lightweight Vision Backbone):**
   - Quantized via static INT8 PTQ (`QOperator`), achieving a **73.7% storage reduction** (`20.91 MB` $\rightarrow$ `5.50 MB`) for ultra-lightweight edge inference.

---

## Final Benchmark Results (200 Validation/Calibration Images, `640x640`, CPU)

| Model | Precision | Size (MB) | Size Reduction | Avg Latency (ms) | Throughput (FPS) | Speedup | Mean Box IoU | AP@50 Consistency | Est. mAP Drop | Target ($\le 2.00\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RT-DETRv2 R18** | FP32 | 77.28 | — | 259.04 | 3.9 | 1.00x | — | — | — | Baseline |
| **RT-DETRv2 R18** | **INT8 (Mixed)** | **46.00** | **-40.5%** | **240.49** | **4.2** | **1.08x** | **99.10%** | **99.44%** | **0.56%** | **Passed** |
| **D-FINE small** | FP32 | 39.58 | — | 231.30 | 4.3 | 1.00x | — | — | — | Baseline |
| **D-FINE small** | **INT8 (Mixed)** | **36.19** | **-8.6%** | **227.98** | **4.4** | **1.01x** | **98.76%** | **99.00%** | **1.00%** | **Passed** |
| **MobileNetV3-Large** | FP32 | 20.91 | — | — | — | — | — | — | — | Baseline |
| **MobileNetV3-Large** | **INT8 (Static)** | **5.50** | **-73.7%** | — | — | — | — | — | — | **Compressed** |

### Deployment Recommendation
- **`RT-DETRv2 R18 (INT8)`** delivers the largest detector memory savings (**-40.5%**, down to `46.00 MB`), strongest CPU acceleration (**1.08x**), and highest detection consistency (**99.44% AP@50**, **0.56% drop**), making it the primary candidate when storage/RAM compression is critical.
- **`D-FINE small (INT8)`** provides the highest overall detector throughput (**4.4 FPS** / `227.98 ms`), smallest detector footprint (`36.19 MB`), and near-identical detection fidelity (**99.00% AP@50 consistency**, **1.00% drop**), making it ideal for latency-sensitive mobile inference.
- **`MobileNetV3-Large (INT8)`** compresses down to **`5.50 MB`** (**-73.7%**), making it ideal for ultra-constrained edge classification/feature-extraction tasks.

---

## Repository Structure

```text
p2-12-model-quantization/
├── .dvc/                               # DVC configuration (Google Drive remote)
├── data/
│   ├── calibration.dvc                 # DVC pointer for calibration dataset
│   └── calibration/                    # 200 representative RGB images (640x640, DVC-tracked)
├── models/
│   ├── onnx_fp32.dvc                   # DVC pointer for FP32 baseline models
│   ├── onnx_int8.dvc                   # DVC pointer for quantized INT8 models
│   ├── onnx_fp32/                      # Exported FP32 ONNX baselines (DVC-tracked)
│   │   ├── dfine_small_fp32.onnx       # 39.58 MB
│   │   ├── mobilenetv3_large_fp32.onnx # 20.91 MB
│   │   └── rtdetrv2_r18_fp32.onnx      # 77.28 MB
│   └── onnx_int8/                      # Quantized INT8 ONNX models (DVC-tracked)
│       ├── dfine_small_int8.onnx       # 36.19 MB
│       ├── mobilenetv3_large_int8.onnx # 5.50 MB
│       └── rtdetrv2_r18_int8.onnx      # 46.00 MB
├── p2-13-Models-DOC/
│   ├── P2-17_Vision_Models_Selection_Report.docx
│   └── placement.json
├── results/
│   └── comparison_report.md            # Detailed FP32 vs INT8 benchmark report
├── scripts/
│   ├── 01_prepare_data.py              # Prepares & resizes calibration dataset
│   ├── 02_export_onnx.py               # Exports PyTorch/HF models to ONNX FP32
│   ├── 03_quantize_int8.py             # Runs Mixed-Precision Static INT8 PTQ
│   └── 04_benchmark.py                 # Evaluates Size, Latency, FPS, IoU & mAP Drop
├── .dvcignore
├── .gitignore
├── requirements.txt
├── setup_dvc.ps1                       # Configures local DVC OAuth credentials & pulls data
└── README.md
```

---

## Quickstart & DVC Artifact Retrieval

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Models & Calibration Data via DVC
The binary `.onnx` models and calibration images are stored on Google Drive and tracked via `.dvc` files. Run the setup script inside `p2-12-model-quantization/` to configure local OAuth credentials and download all artifacts automatically:
```powershell
.\setup_dvc.ps1
```

### 3. Run Pipeline from Scratch (Optional)
```bash
python scripts/01_prepare_data.py
python scripts/02_export_onnx.py
python scripts/03_quantize_int8.py
python scripts/04_benchmark.py
```

