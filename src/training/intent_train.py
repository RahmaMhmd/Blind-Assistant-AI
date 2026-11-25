import json
from pathlib import Path

from .intent_data import ALLOWED_INTENTS
from .train_utils import log_training_run, pick_device

MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da"
OUTPUT_REL = Path("models/checkpoints/camelbert_intent")


def train_intent(
    train_path: Path,
    root: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_len: int = 48,
    seed: int = 42,
    track: bool = True,
) -> Path:
    """fine tune camelbert on 8 app intents from train jsonl"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = pick_device()
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label2id = {name: idx for idx, name in enumerate(ALLOWED_INTENTS)}
    id2label = {idx: name for idx, name in enumerate(ALLOWED_INTENTS)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    labels = torch.tensor([label2id[r["intent"]] for r in rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels), batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(ALLOWED_INTENTS), id2label=id2label, label2id=label2id
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for ids, mask, y in loader:
            opt.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            opt.step()

    out_dir = root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if track:
        log_training_run(root, "train-camelbert-intent", {"epochs": epochs, "lr": lr, "batch": batch_size, "samples": len(rows)}, {})
    return out_dir
