| Feature | Status | Primary | Fallback | Confirm in |
|---|---|---|---|---|
| currency_recognition | tentative | dfine-small-egp-currency-int8 (local, Apache-2.0, 36.19 MB) | mobilenetv3-large-egp-int8 (local, Apache-2.0, 5.5 MB) | P4-06 |
| depth_estimation | tentative | depth-anything-v2-small (local, Apache-2.0, 94.6 MB) | depth-anything-v2-small-cloud (cloud, Apache-2.0, 94.6 MB) | P4-05 |
| image_quality | tentative | opencv-laplacian-brightness (local, Apache-2.0, 0.0 MB) | mobilenetv3-quality-classifier (local, BSD-3-Clause, 9.8 MB) | P4-08 |
| keyword_spotter | tentative | sherpa-onnx-kws (local, Apache-2.0, 5.0 MB) | silero-vad (local, MIT, 10.4 MB) | P4-01 |
| object_detection | tentative | rtdetrv2-r18-int8 (local, Apache-2.0, 46.0 MB) | dfine-small-int8 (local, Apache-2.0, 36.19 MB) | P4-02 |
| object_tracking | tentative | botsort-cmc-orb (local, MIT, 0.0 MB) | bytetrack (local, MIT, 0.0 MB) | P4-05 |
| obstacle_detection | tentative | dfine-small-obstacles-int8 (local, Apache-2.0, 36.19 MB) | efficientdet-lite0-int8 (local, Apache-2.0, 4.4 MB) | P4-05 |
| ocr | tentative | paddleocr-ppocrv5-arabic-mobile (local, Apache-2.0, 12.8 MB) | paddleocr-vl (cloud, Apache-2.0, 2045.0 MB) | P4-03 |
| qr_barcode | tentative | ml-kit-barcode-scanning (local, Google APIs Terms of Service, 2.4 MB) | zxing-cpp (local, Apache-2.0, 1.1 MB) | P7-09 |
| scene_description | tentative | qwen2.5-vl-7b-instruct (cloud, Apache-2.0, 15816.1 MB) | internvl3.5-8b (cloud, Apache-2.0, 16266.6 MB) | P4-07 |
| speech_recognition | tentative | whisper-large-v3-turbo (cloud, MIT, 1542.9 MB) | whisper-large-v3-arabic (cloud, MIT, 5888.3 MB) | P4-01 |
| voice_command | tentative | camelbert-da-intent (cloud, Apache-2.0, 418.5 MB) | qwen3-1.7b-instruct (cloud, Apache-2.0, 3875.3 MB) | P4-01 |

Considered

- currency_recognition: yolov8-currency rejected - AGPL-3.0 license blocks commercial shipping
- depth_estimation: apple-depth-pro deferred - 1B params too heavy and custom Apple license unconfirmed
- image_quality: pyiqa-musiq-maniqa research_only - cloud comparison only and not inside the app
- object_detection: yolo11-yolo26 rejected - AGPL-3.0 license blocks commercial shipping
- object_detection: rf-detr deferred - check phone speed after rtdetrv2 and dfine
- object_tracking: cotracker3 research_only - CC-BY-NC-4.0 blocks commercial use
- obstacle_detection: ssd-mobilenetv2 deferred - lower accuracy than efficientdet lite0
- ocr: ml-kit-text-recognition-v2 deferred - Arabic is not in the supported experimental or mapped lists on 2026-11-21 so it is English only. keep as optional English path and test it in P2-13
- ocr: tesseract-5 baseline - baseline for CER and WER only
- ocr: qari-ocr deferred - 2B model on Qwen2-VL with Apache-2.0 verified on NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct. second cloud candidate to compare on our data
- ocr: baseer research_only - 3B model on Qwen2.5-VL-3B. base repo is gated and public variant Misraj/Baseer__Nakba is CC-BY-NC-SA-4.0 non-commercial
- ocr: surya research_only - 650M model. code is Apache-2.0 and weights use modified AI Pubs OpenRAIL-M v0.1 with a 5M USD annual revenue or funding cap
- qr_barcode: opencv-qrcode-detector deferred - compare on our images before dropping it
- scene_description: moondream2 baseline - Apache-2.0 1.9B small fast VLM baseline
- scene_description: florence-2-large deferred - MIT weak Arabic output
- scene_description: smolvlm2-500m deferred - Apache-2.0 too small for good Arabic descriptions
- speech_recognition: whisper-small-eg baseline - Apache-2.0 small 241.7M params baseline on Hugging Face
- speech_recognition: mms-1b-all rejected - CC-BY-NC-4.0 blocks commercial use
- speech_recognition: canary-1b rejected - CC-BY-NC-4.0 blocks commercial use
- voice_command: gliner-multi-v2.1 baseline - Apache-2.0 zero shot slot baseline before fine tuning
- voice_command: marbert-v2 deferred - license tag not confirmed on Hugging Face card
