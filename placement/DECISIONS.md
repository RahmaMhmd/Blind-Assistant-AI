| Feature | Status | Primary | Fallback | Confirm in |
|---|---|---|---|---|
| image_quality | tentative | opencv-laplacian-brightness (local, Apache-2.0, 0.0 MB) | mobilenetv3-quality-classifier (local, BSD-3-Clause, 9.8 MB) | P4-08 |
| ocr | tentative | paddleocr-ppocrv5-arabic-mobile (local, Apache-2.0, 12.8 MB) | paddleocr-vl (cloud, Apache-2.0, 2045.0 MB) | P4-03 |
| qr_barcode | tentative | ml-kit-barcode-scanning (local, Google APIs Terms of Service, 2.4 MB) | zxing-cpp (local, Apache-2.0, 1.1 MB) | P7-09 |

Considered

- image_quality: pyiqa-musiq-maniqa research_only - cloud comparison only and not inside the app
- ocr: ml-kit-text-recognition-v2 deferred - Arabic is not in the supported experimental or mapped lists on 2026-11-21 so it is English only. keep as optional English path and test it in P2-13
- ocr: tesseract-5 baseline - baseline for CER and WER only
- ocr: qari-ocr deferred - 2B model on Qwen2-VL with Apache-2.0 verified on NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct. second cloud candidate to compare on our data
- ocr: baseer research_only - 3B model on Qwen2.5-VL-3B. base repo is gated and public variant Misraj/Baseer__Nakba is CC-BY-NC-SA-4.0 non-commercial
- ocr: surya research_only - 650M model. code is Apache-2.0 and weights use modified AI Pubs OpenRAIL-M v0.1 with a 5M USD annual revenue or funding cap
- qr_barcode: opencv-qrcode-detector deferred - compare on our images before dropping it
