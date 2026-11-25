# Blind Assistant AI

Backend inference service, evaluation harness, INT8 quantization pipeline, dataset CLI, placement manifest, and Android phone benchmark harness.

## 1. Environment Setup

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Split files are also available (`requirements-service.txt`, `requirements-vision.txt`, `requirements-test.txt`, `requirements-eval.txt`).

## 2. Service and Tests

Run the FastAPI service (fixture handlers in Phase 2; `requirements-vision.txt` and `paddleocr` join Docker in `P4-10`):

```bash
python -m src.service
uvicorn src.service.main:app --reload
```

Run the test suite:

```bash
pytest -q
```

## 3. Training

```bash
python -m src.training currency-calib
python -m src.training currency
python -m src.training intent-data --csv data/templates/voice_intents_template.csv
python -m src.training intent --train data/intent/train.jsonl
```

## 4. Data, Evaluation, Quantization, and Placement

```bash
python -m data_tools check ocr-text
python -m src.evaluation --model paddleocr-arabic-rec --testset ocr_eval --track
python -m src.quantization calibrate
python -m src.quantization export --model dfine_small
python -m src.quantization quantize --model dfine_small
python -m src.quantization benchmark --model dfine_small --track
python -m src.placement build
python -m src.placement check
```

## 5. Android Benchmark, MLflow UI, and DVC

```bash
cd android/benchmark && ./gradlew test
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
dvc pull
```

## 6. Backlog Scope and Caveats

- Official Phase 2 tasks span `P2-01` through `P2-16`; `P2-17` and `P2-18` are Phase 2 closeout extensions, while `OcrHandler` (`P4-10`), `imagequality` (`P4-08`), and QR tests (`P7-09`) are early pre-work.
- See `placement/SDD_ALIGNMENT.md` for SDD deviations (`PaddleOCR` vs `ML Kit`, `ZXing` safety, `1.5s` timeout risk) and dataset caveats (seed data limits, single-device Xiaomi baseline, currency student collection license blocker).
