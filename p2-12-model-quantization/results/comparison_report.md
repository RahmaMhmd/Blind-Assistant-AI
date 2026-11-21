# Model Quantization Comparison Report (FP32 vs. Static INT8)

## 1. Executive Summary
This report presents the post-training quantization (PTQ) evaluation for two real-time transformer-based object detection models (**RT-DETRv2 R18** and **D-FINE small**) targeting edge/mobile deployment. Using **ONNX Runtime Static INT8 Quantization (QOperator)** with **MinMax calibration** across **200 representative COCO validation images (`640x640`)**, both models successfully satisfied the accuracy preservation target of **Estimated mAP Drop $\le 2.00\%$** while achieving model compression and CPU inference speedups.

---

## 2. Quantitative Benchmark Comparison

| Model | Precision | Quantization Strategy | Size (MB) | Size Reduction | Avg Latency (ms) | Throughput (FPS) | Speedup | Mean Box IoU | AP@50 Consistency | Est. mAP Drop | Target ($\le 2.00\%$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RT-DETRv2 R18** | FP32 | Baseline (Unquantized) | 77.28 | — | 259.04 | 3.9 | 1.00x | — | — | — | Baseline |
| **RT-DETRv2 R18** | **INT8** | Mixed-Precision Static INT8 (13 Deep Backbone Convs) | **46.00** | **-40.5%** | **240.49** | **4.2** | **1.08x** | **99.10%** | **99.44%** | **0.56%** | **Passed** |
| **D-FINE small** | FP32 | Baseline (Unquantized) | 39.58 | — | 231.30 | 4.3 | 1.00x | — | — | — | Baseline |
| **D-FINE small** | **INT8** | Mixed-Precision Static INT8 (`stages.3[:-1]` + `stages.2` 1x1) | **36.19** | **-8.6%** | **227.98** | **4.4** | **1.01x** | **98.76%** | **99.00%** | **1.00%** | **Passed** |

---

## 3. Architectural Analysis & Engineering Findings

### 3.1 RT-DETRv2 R18 (ResNet-18 Backbone)
- **Compression & Speedup:** Because ResNet-18 relies heavily on standard dense $3\times3$ convolutions, static INT8 quantization across the 13 deep backbone layers (`stages.1` through `stages.3`) compressed the model by **40.5%** (saving **31.28 MB**) and improved CPU throughput by **1.08x** (`240.49 ms` / `4.2 FPS`).
- **Accuracy Preservation:** Preserving the early `embedder`/`stage-0` stem, Hybrid Encoder, and Transformer Decoder layers in FP32 limited the estimated mAP drop to just **0.56%** (**99.44%** AP@50 consistency and **99.10%** bounding-box IoU overlap with the FP32 baseline).

### 3.2 D-FINE small (HGNetv2 Backbone + FDR Heads)
- **Architectural Sensitivity:** `D-FINE small` is already compact at baseline (`39.58 MB`) and utilizes `HGNetv2` (depthwise separable convolutions) paired with Fine-Grained Distribution Refinement (FDR) heads. Full-graph quantization caused severe accuracy degradation due to sensitivity in the FDR probability distributions and the final backbone-to-encoder transition layer.
- **Selective Mixed-Precision Solution:** Quantizing the deep standard convolutions in `stages.3[:-1]` (preserving the final encoder-interface convolution in FP32) combined with the $1\times1$ pointwise convolutions in `stages.2` (14 nodes total) reduced model size by **8.6%** (`36.19 MB`), improved CPU inference speed to **1.01x** (`227.98 ms` / `4.4 FPS`), and kept the estimated mAP drop at just **1.00%** (**99.00%** AP@50 consistency).

---

## 4. Deployment Recommendation
1. **Primary Mobile INT8 Candidate — `RT-DETRv2 R18 (INT8)` (`models/onnx_int8/rtdetrv2_r18_int8.onnx`):** Delivers the highest storage and memory bandwidth reduction (**-40.5%**), strongest CPU acceleration (**1.08x**), and highest detection fidelity (**99.44%** AP@50 consistency, **0.56%** mAP drop).
2. **Ultra-Low Latency / Lightweight Candidate — `D-FINE small (INT8)` (`models/onnx_int8/dfine_small_int8.onnx`):** Achieves the lowest absolute latency (**227.98 ms / 4.4 FPS**), smallest footprint (**36.19 MB**), and near-identical detection fidelity (**1.00%** mAP drop).