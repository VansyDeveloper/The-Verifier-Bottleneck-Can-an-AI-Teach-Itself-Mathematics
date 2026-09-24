"""Check the published per-set table against independently audited summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HEADING = "## Сводная таблица проверенных наборов"
SEEDS = ("85000", "85001", "85002")


def _number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _summaries(data_root: Path) -> dict[tuple[int, int], dict]:
    summaries: dict[tuple[int, int], dict] = {}
    for path in sorted(data_root.glob("*/analysis_summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item["k"], item["subset"])
        if key in summaries:
            raise ValueError(f"duplicate audited summary for {key}")
        if item["status"] != "VALID":
            raise ValueError(f"not VALID: {path}")
        if item["completed_trainings"] != 6 or item["completed_evaluations"] != 6:
            raise ValueError(f"not fully accepted: {path}")
        if item["withheld_hit32_difference"]["n"] != 3:
            raise ValueError(f"not three paired seeds: {path}")
        if set(item["per_seed_differences"]) != set(SEEDS):
            raise ValueError(f"wrong paired seed set: {path}")
        summaries[key] = item
    return summaries


def _report_rows(report: Path) -> dict[tuple[int, int], list[str]]:
    lines = report.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(HEADING) + 1
    except ValueError as exc:
        raise ValueError("missing audited-set table heading") from exc
    rows: dict[tuple[int, int], list[str]] = {}
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            raise ValueError(f"wrong table width: {line}")
        if cells[0] == "Набор" or cells[0].startswith("---"):
            continue
        parts = cells[0].split("/")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError(f"invalid set key: {cells[0]}")
        key = (int(parts[0]), int(parts[1]))
        if key in rows:
            raise ValueError(f"duplicate report row: {key}")
        rows[key] = cells
    return rows


def validate_report_table(report: Path, data_root: Path) -> int:
    summaries = _summaries(data_root)
    rows = _report_rows(report)
    missing = sorted(set(summaries) - set(rows))
    extra = sorted(set(rows) - set(summaries))
    if missing or extra:
        raise ValueError(f"missing audited sets {missing}; extra report rows {extra}")
    for key, item in summaries.items():
        cells = rows[key]
        stats = item["withheld_hit32_difference"]
        expected = {
            "per_seed_differences": " / ".join(
                _number(item["per_seed_differences"][seed], 3) for seed in SEEDS
            ),
            "mean_difference": _number(stats["mean_difference"], 4),
            "sd": _number(stats["sd"], 4),
            "ci95": (
                f"от {_number(stats['ci95_low'], 4)} "
                f"до {_number(stats['ci95_high'], 4)}"
            ),
            "exact_signflip_two_sided_p": _number(
                stats["exact_signflip_two_sided_p"], 2
            ),
        }
        actual = dict(zip(expected, (cells[2], cells[3], cells[4], cells[5], cells[6])))
        for field, want in expected.items():
            if actual[field] != want:
                raise ValueError(
                    f"{key} {field}: report={actual[field]!r}, audited={want!r}"
                )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = validate_report_table(args.report, args.data_root)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"REPORT_TABLE_VALID {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
