from .models import ModelRef, PlacementFile

HEADER = "| Feature | Status | Primary | Fallback | Confirm in |\n|---|---|---|---|---|"


def _cell(ref: ModelRef | None) -> str:
    if ref is None:
        return "-"
    size = "?" if ref.size_mb is None else f"{ref.size_mb} MB"

    return f"{ref.name} ({ref.placement.value}, {ref.license.name}, {size})"


def render_table(placement: PlacementFile) -> str:
    rows = [HEADER]
    notes = []

    for name, entry in placement.features.items():
        confirm = entry.confirm_in or "-"
        row = [
            name,
            entry.status.value,
            _cell(entry.primary),
            _cell(entry.fallback),
            confirm,
        ]
        rows.append("| " + " | ".join(row) + " |")
        notes += [
            f"- {name}: {c.name} {c.outcome.value} - {c.reason}"
            for c in entry.considered
        ]

    return "\n".join([*rows, "", "Considered", "", *notes]) + "\n"
