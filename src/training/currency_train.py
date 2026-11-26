import copy
import time
from pathlib import Path

from .currency_data import DENOMINATIONS, load_currency_splits
from .train_utils import eval_classifier, log_training_run, make_image_loader, pick_device

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CHECKPOINT_REL = Path("models/checkpoints/mobilenetv3_large_egp.pth")


def train_currency(
    root: Path, epochs: int = 10, batch_size: int = 32, lr: float = 3e-4, seed: int = 42, track: bool = True
) -> dict:
    """fine tune mobilenetv3 large on currency split and log metrics"""
    import timm
    import torch
    import torchvision.transforms as T

    torch.manual_seed(seed)
    device = pick_device()
    tr_rows, va_rows, te_rows = load_currency_splits(root, seed=seed)
    tr_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.ColorJitter(0.3, 0.3, 0.3),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    ev_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr_ld = make_image_loader(tr_rows, tr_tf, batch_size, True)
    va_ld = make_image_loader(va_rows, ev_tf, batch_size, False)
    te_ld = make_image_loader(te_rows, ev_tf, batch_size, False)

    model = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=True, num_classes=len(DENOMINATIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val, best_state, target_epochs = -1.0, copy.deepcopy(model.state_dict()), epochs

    ep = 1
    while ep <= target_epochs:
        t0 = time.perf_counter()
        model.train()
        for x, y in tr_ld:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        val_acc, _, _, _ = eval_classifier(model, va_ld, device, len(DENOMINATIONS))
        elapsed = time.perf_counter() - t0
        if ep == 1 and epochs == 10 and elapsed > 240.0:
            target_epochs = 6
        print(f"epoch {ep}/{target_epochs} val_acc={val_acc:.4f} elapsed_s={elapsed:.1f}", flush=True)
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
        ep += 1

    model.load_state_dict(best_state)
    ckpt = root / CHECKPOINT_REL
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt)
    te_acc, te_f1, recalls, conf = eval_classifier(model, te_ld, device, len(DENOMINATIONS))
    metrics = {"val_accuracy": round(best_val, 4), "test_accuracy": round(te_acc, 4), "test_macro_f1": round(te_f1, 4)}
    for name, rec in zip(DENOMINATIONS, recalls):
        metrics[f"recall_{name}"] = round(rec, 4)
    print("confusion_matrix:", conf, flush=True)
    if track:
        params = {"epochs": target_epochs, "lr": lr, "batch": batch_size, "device": device, "train": len(tr_rows), "val": len(va_rows), "test": len(te_rows)}
        metrics["mlflow_run_id"] = log_training_run(root, "train-mobilenetv3-egp", params, metrics)
    return {"metrics": metrics, "confusion": conf}
