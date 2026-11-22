# Data workflow

Proposed capture protocol. Numbers and names need team agreement before the first batch.

## File name

`<dataset>_<batch>_<group>_<seq>.<ext>` for example `ocr-text_b01_g014_007.jpg`

- dataset is one of the names in `data/datasets.json`
- batch is one capture session `b01` `b02`
- group is one physical object or scene and every photo of it shares the group id
- group ids are never reused inside a dataset
- the test split is chosen by hashing the group id so one object never lands in both splits

## Commands

Run from the repo root.

```
python -m data_tools card <dataset>
python -m data_tools ingest <dataset> <incoming_dir> --face-model <yunet.onnx>
python -m data_tools log <dataset> --who <name> --action collected --batch b01 --count 30
python -m data_tools lock <dataset>
python -m data_tools check <dataset> --face-model <yunet.onnx>
python -m data_tools publish <dataset> --push
```

Order for each batch: ingest, fill the manifest in `data/manifests`, log collected labeled reviewed, lock, check, publish.
The reviewer must be a different person from the labeler.

## Where things live

- images and audio: `data/raw/<dataset>` tracked by DVC
- labels: `data/manifests/<dataset>.csv` in git so labels are reviewed in pull requests
- test lock: `data/locks/<dataset>.json` in git and append only
- who did what and when: `data/ledger/<dataset>.jsonl` in git
- data card: `data/cards/<dataset>.md` in git
