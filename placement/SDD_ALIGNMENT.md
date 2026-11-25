# SDD and Backlog Alignment ADR

Decision record reconciling Phase 2 experimental results with the SDD and Backlog.

## 1. Backlog Scope Alignment

- Official Phase 2 covers `P2-01` through `P2-16`.
- `P2-17` (model selection) and `P2-18` (placement manifest) are Phase 2 closeout extensions beyond the original 16-task sheet.
- Early pre-work in this repo belongs to later phases: `OcrHandler` and `paddle_engine.py` (`P4-10`), `imagequality` calibration (`P4-08`), and QR comparison (`P7-09`).
- All `confirmIn` fields in `placement/features/*.json` map to official backlog tasks:
  - Speech: `P3-09` (`speech_recognition`), `P3-10` (`keyword_spotter`), `P3-11` (`voice_command`)
  - Text and Quality: `P4-08` (`image_quality`), `P4-09` (`ocr`)
  - Vision and Currency: `P5-09` (`object_detection`), `P5-10` (`scene_description`), `P5-12` (`currency_recognition`)
  - Obstacles and Depth: `P6-09` (`obstacle_detection`), `P6-10` (`depth_estimation`), `P6-11` (`object_tracking`)
  - Codes: `P7-09` (`qr_barcode`)

## 2. SDD Architecture Deviations Requiring ADR Update

1. **OCR candidate (ML Kit to PaddleOCR)**: On 60 real Arabic crops on Xiaomi `2410FPCC5G`, ML Kit scored `0.0%` (`0/60`) vs `83.3%` English (`50/60`). `PaddleOCR PP-OCRv5` scored `0.0313` CER (`0.1138` WER, `82.6 ms` CPU) across 120 `ar+en` crops. SDD must update OCR from ML Kit to PaddleOCR.
2. **QR safety (ML Kit to ZXing)**: ML Kit scored `54.12%` correct and `25.49%` wrong on 255 real non-test images (`35.2 ms`). Speaking unverified wrong codes is unsafe for blind users. `zxing-cpp` scored `75.29%` (`82.61%` on 46 locked test, `1.57%` wrong, `2.48 ms` CPU). `P7-09` must enforce multi-frame consensus before TTS output.
3. **1.5s local router timeout vs D-FINE CPU latency**: `dfine-small-int8` measured `1.45s` mean and `1.63s` p95 cool (`2.65s` warm) on Xiaomi CPU, risking frequent timeouts against the SDD `1.5s` local router limit. `obstacle_detection` keeps `efficientdet-lite0-int8` (`litert`, `4.4 MB`) as local primary; `object_detection` needs NNAPI or GPU delegate testing or cloud fallback in `P5-09`.
4. **Currency classifier matching SDD**: Switched `currency_recognition` primary to `mobilenetv3-large-egp-int8` (`28.5 ms` CPU, `5.69 MB`, `baseline_only`) matching the SDD classifier design, with `dfine-small-egp-currency-int8` as cloud `baseline_only` fallback until `P5-12` training.

## 3. Honest Phase 2 Data and Benchmark Limits

- **Single phone baseline**: Measured only on Xiaomi `2410FPCC5G` (`Android 16`, `2.9 GB` free RAM). `PaddleOCR`, `Sherpa-ONNX KWS`, and `Depth Anything V2` are not benchmarked on phone yet.
- **Public seed datasets**: `ocr-text` uses public signage (including Moroccan EvArEST words), `voice-commands` has `GENERAL_SPEECH` audio only (zero real command recordings yet for `P3-10` and `P3-11`), and `image-quality` was calibrated on public scene-text crops rather than real mobile camera photos (`P4-08`).
- **Release blocker**: `currency` student collection license is unverified and must be verified or replaced before commercial release (`P5-12`).
- **Model registry**: Three quantized ONNX models carry real `sha256`, `version`, and `rollbackTarget`; remaining features stay `tentative` until trained and exported in Phases 3-7.
