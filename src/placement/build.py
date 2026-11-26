import json
from pathlib import Path

from .checks import entry_problems
from .models import FeatureEntry, PlacementFile

TEXT_ENCODING = "utf-8"
PLACEMENT_DIR = Path("placement")
SOURCE_DIR = PLACEMENT_DIR / "features"
BUILT_FILE = PLACEMENT_DIR / "placement.json"
SCHEMA_FILE = PLACEMENT_DIR / "placement.schema.json"
TABLE_FILE = PLACEMENT_DIR / "DECISIONS.md"
COMPACT = (",", ":")


class PlacementInvalid(Exception):
    pass


def load_sources(root: Path = Path(".")) -> PlacementFile:
    features = {}

    for path in sorted((root / SOURCE_DIR).glob("*.json")):
        raw = json.loads(path.read_text(encoding=TEXT_ENCODING))
        features[path.stem] = FeatureEntry.model_validate(raw)

    problems = [p for name, e in features.items() for p in entry_problems(name, e)]

    if problems:
        raise PlacementInvalid("; ".join(problems))

    return PlacementFile(features=features)


def _compact(entry: dict) -> str:
    return json.dumps(entry, separators=COMPACT, ensure_ascii=False)


def dump_built(placement: PlacementFile) -> str:
    """one feature per line so the file stays short"""
    data = placement.model_dump(mode="json", by_alias=True)
    rows = [
        f"{json.dumps(name)}:{_compact(entry)}"
        for name, entry in data["features"].items()
    ]
    head = f'{{"schemaVersion":{json.dumps(data["schemaVersion"])},"features":{{\n'

    return head + ",\n".join(rows) + "\n}}\n"


def dump_schema() -> str:
    schema = PlacementFile.model_json_schema(by_alias=True)

    return json.dumps(schema, separators=COMPACT) + "\n"
