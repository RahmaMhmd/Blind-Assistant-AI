# Blind Assistant AI Repository

## 1. Clone Repository

```powershell
git clone -b main https://github.com/RahmaMhmd/Blind-Assistant-AI.git
cd Blind-Assistant-AI
```

## 2. Create Environment & Install Libraries

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. MLflow

Run MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

http://127.0.0.1:5000

## 4. DVC – Google Drive Setup

Check the remote:

```powershell
dvc remote list
```

The remote should be named `gdrive`.

The Google Drive folder ID is already configured in `.dvc/config`.

If the remote is missing, add it using:

```powershell
dvc remote add -d gdrive gdrive://GOOGLE_DRIVE_FOLDER_ID/dvcstore
```

Configure OAuth credentials locally:

```powershell
dvc config --local remote.gdrive.gdrive_client_id "CLIENT_ID"
dvc config --local remote.gdrive.gdrive_client_secret "CLIENT_SECRET"
```

## 5. Download Existing Data

```powershell
dvc pull
```

## 6. Upload New or Updated Data

Track the dataset:

```powershell
dvc add data/raw/DATASET_NAME
```

Upload data to Google Drive:

```powershell
dvc push
```

---

# Baseer Unified Contract - AI Service

This package defines the shared request and response contracts used by the Baseer AI Service.

The contract is designed to maintain compatibility with the current Baseer Backend API while preserving important AI-specific tracking information.

## 7. Project Structure

```text
src/
    contracts/
        __init__.py
        inference.py
        errors.py
        validators.py

tests/
    test_inference_contract.py
```

## 8. Supported AI Features

The current contract supports:

- OCR
- Voice Command / Intent Recognition
- Scene Description
- Currency Recognition
- Color Detection
- Traffic Sign Recognition
- Face Expression Recognition
- Obstacle Detection

QR Navigation is not included as an AI inference feature because route calculation is currently handled by the Backend navigation module.

## 9. Input Contract

### Image Input

Image-based features receive JSON containing:

```json
{
    "imageBase64": "data:image/jpeg;base64,..."
}
```

Supported formats:

- JPEG
- PNG

The maximum image size is currently 10 MiB (provisional).

### Voice Command Input

Voice command currently receives transcribed text:

```json
{
    "userSpeech": "اقرأ الورقة دي"
}
```

The current Backend OpenAPI does not define a raw audio upload endpoint for Speech-to-Text.

## 10. Success Response

All successful AI responses contain:

- success
- statusCode
- timestamp
- requestId
- modelName
- processingTimeMs
- data

Example:

```json
{
    "success": true,
    "statusCode": 200,
    "timestamp": "2026-11-10T14:30:00Z",
    "requestId": "req-12345",
    "modelName": "PaddleOCR",
    "processingTimeMs": 245.6,
    "data": {
        "extractedText": "أهلاً",
        "confidence": 0.95
    }
}
```

### AI Tracking Fields

#### requestId

Unique identifier for each inference request.

Used for:

- Request tracking
- Debugging
- Log correlation
- Error tracking

#### modelName

Name of the AI model that processed the request.

Used for:

- Model tracking
- Comparing different models
- Experiment analysis
- Debugging model behavior

#### processingTimeMs

Time taken by the AI model/service to process the request, measured in milliseconds.

Used for:

- Performance monitoring
- Latency measurement
- Model comparison
- Optimization

## 11. Error Response

All errors contain:

- success
- statusCode
- errorCode
- message
- details
- timestamp
- requestId
- modelName
- processingTimeMs

Example:

```json
{
    "success": false,
    "statusCode": 422,
    "errorCode": "ERR_IMAGE_BLURRY",
    "message": "الصورة مشوشة أو غير واضحة، يرجى إعادة التقاطها بثبات",
    "details": [],
    "timestamp": "2026-11-10T14:30:00Z",
    "requestId": "req-12345",
    "modelName": "PaddleOCR",
    "processingTimeMs": 120.5
}
```

`modelName` and `processingTimeMs` may be null when an error occurs before model inference starts.

`requestId` is required for tracking every request.

## 12. Arabic Error Messages

User-facing error messages are generated directly in Arabic inside the AI Service.

The Backend does not need to translate English error messages.

Error codes remain in English for stable programmatic handling.

## 13. Response Compatibility

The response preserves the main Baseer Backend envelope:

- success
- statusCode
- timestamp
- data

The following AI-specific fields are additionally included:

- requestId
- modelName
- processingTimeMs
## 15. Run Tests

Activate the project's virtual environment, then run:

```powershell
pytest -v
```