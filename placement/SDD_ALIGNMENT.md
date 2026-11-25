# SDD and Backlog Alignment ADR

Decision record reconciling Phase 2 results with the SDD and Backlog.

## 1. Backlog Scope Alignment

- Official Phase 2 covers `P2-01` through `P2-16`; `P2-17` and `P2-18` are Phase 2 closeout extensions, while `OcrHandler` (`P4-10`), `imagequality` (`P4-08`), and QR tests (`P7-09`) are early pre-work.
- `confirmIn` mapping: Speech `P3-09`/`P3-10`/`P3-11`; Text and Quality `P4-08`/`P4-09`; Vision and Currency `P5-09`/`P5-10`/`P5-12`; Obstacles and Depth `P6-09`/`P6-10`/`P6-11`; Codes `P7-09`.

## 2. SDD Architecture Deviations

1. **OCR (`ML Kit` to `PaddleOCR`)**: On 60 real Arabic crops on Xiaomi `2410FPCC5G`, ML Kit scored `0.0%` (`0/60`) vs `83.3%` English (`50/60`). `PaddleOCR PP-OCRv5` scored `0.0313` CER (`0.1138` WER, `82.6 ms` CPU) across 120 `ar+en` crops.
2. **QR safety (`ML Kit` to `ZXing`)**: ML Kit scored `54.12%` correct and `25.49%` wrong on 255 real images (`35.2 ms`), unsafe without consensus. `zxing-cpp` scored `75.29%` (`82.61%` locked test, `1.57%` wrong, `2.48 ms`). `P7-09` requires multi-frame verification.
3. **1.5s local timeout vs D-FINE vs EfficientDet**: `dfine-small-int8` measured `1.45s` mean / `1.63s` p95 cool (`1.96s` p95 run 2, `2.65s` warm) on Xiaomi CPU, risking the SDD `1.5s` limit. `obstacle_detection` (`P6-09`) uses `efficientdet-lite0-int8` (`litert`, `4.39 MB`, `100.5 ms` median / `101.2 ms` p95 zero-tensor on run 2). `object_detection` (`P5-09`) needs NNAPI/GPU or cloud fallback.
4. **Currency classifier (`P5-12`)**: Fine-tuned `mobilenetv3-large-egp` hit `96.90%` FP32 accuracy on locked `currency_test` (`451` images). `percentile` INT8 (`4.46 MB`, `31.8 ms` median / `35.2 ms` p95 on Xiaomi) achieved `0.90` agreement and `82.93%` test accuracy (`entropy` and `minmax` scored `0.60` / `47.01%`), replacing the old `0.40`/`0.62` untrained ImageNet COCO numbers. `usage` stays `baseline_only` until data license clearance.

## 3. Evaluation Limits

- **Small test sets**: `coco_det` (`60`), `egyptian_asr` (`60`), `ocr_eval` (`120` crops). Gaps under ~5 points (e.g., `D-FINE` vs `RT-DETRv2` at `3.3 mAP50`) lie inside sampling noise and do not drive decisions; CPU latency drives placement.
- **OCR crops**: Measured on cropped words (recognition only); full sign detection plus recognition is unmeasured so `3.13%` CER is optimistic.
- **Image quality**: Calibrated on TRAIN (`346`), tested on locked TEST (`104`) in `experiments/commands.py::quality_calibrate`. `82.35%` sharp recall rejects ~18% of sharp photos; `P4-08` must raise sharp recall (higher weight or soft hold-steady prompt instead of hard reject).
- **Seed data and invalid intent run**: `ocr-text` includes Moroccan EvArEST signage; `voice-commands` has `GENERAL_SPEECH` only; `image-quality` used public scene text; `currency` student collection license is a commercial release blocker. The MASSIVE intent run (`0.0%` on mismatched labels with untrained head) was soft-deleted from MLflow.
- **Single phone and variance**: Run only on one Xiaomi `2410FPCC5G`, showing run-to-run CPU variance (e.g., `mobilenetv3_large_int8` `28.5` vs `48.7 ms`; `dfine` p95 `1.63` vs `1.96s`). `PaddleOCR`, `Sherpa-ONNX`, and `Depth Anything V2` are not phone-benchmarked yet.
- **ASR evaluation note**: whisper-large-v3-turbo weights downloaded but full 60 clip CPU eval exceeded 60 min budget so the chosen primary is not yet evaluated; whisper-small-eg 71.06% WER is an unchosen small baseline and does not evaluate the primary; evaluate in P3-09 on cloud GPU.
