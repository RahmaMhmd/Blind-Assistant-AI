# Blind Assistant AI – Team Setup

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
