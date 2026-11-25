from pathlib import Path


def pick_device() -> str:
    """choose cuda only when a small kernel runs"""
    import torch

    if torch.cuda.is_available():
        try:
            a = torch.zeros(2, 2, device="cuda")
            _ = (a @ a).sum().item()
            return "cuda"
        except Exception:
            pass
    return "cpu"


def make_image_loader(rows, transform, batch_size: int, shuffle: bool):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class ImageRowsDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, idx: int):
            path, label, _ = rows[idx]
            with Image.open(path) as img:
                return transform(img.convert("RGB")), label

    return DataLoader(ImageRowsDataset(), batch_size=batch_size, shuffle=shuffle)


def eval_classifier(model, loader, device: str, n_cls: int):
    import torch

    model.eval()
    conf = [[0] * n_cls for _ in range(n_cls)]
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().tolist()
            for t, p in zip(y.tolist(), preds):
                conf[t][p] += 1
    total = sum(sum(r) for r in conf)
    acc = sum(conf[i][i] for i in range(n_cls)) / total if total else 0.0
    recalls, f1s = [], []
    for c in range(n_cls):
        tp = conf[c][c]
        fn = sum(conf[c]) - tp
        fp = sum(conf[r][c] for r in range(n_cls)) - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        recalls.append(rec)
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)
    return acc, sum(f1s) / n_cls, recalls, conf


def log_training_run(root: Path, run_name: str, params: dict, metrics: dict) -> str:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("Blind Assistant Experiments")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "training")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return run.info.run_id
