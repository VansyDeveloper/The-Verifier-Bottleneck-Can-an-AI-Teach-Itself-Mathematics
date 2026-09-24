"""Regression tests for the final report's audited-set table."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_report.py")
TABLE = """# Отчёт

## Сводная таблица проверенных наборов

| Набор | Сервер, GPU | Разности по трём seed | Средняя | SD | 95% t-интервал | Точное p |
|---|---|---|---:|---:|---|---:|
| 2/2 | ccmplanner, V100 | -0,196 / -0,224 / -0,246 | -0,2220 | 0,0251 | от -0,2843 до -0,1597 | 0,25 |

## Следующий раздел
"""


def run_check(tmp_path: Path, table: str, *, include_summary: bool = True):
    report = tmp_path / "REPORT.md"
    report.write_text(table, encoding="utf-8")
    data = tmp_path / "data"
    if include_summary:
        summary_dir = data / "k2_s2"
        summary_dir.mkdir(parents=True)
        summary = {
            "status": "VALID",
            "k": 2,
            "subset": 2,
            "completed_trainings": 6,
            "completed_evaluations": 6,
            "withheld_hit32_difference": {
                "n": 3,
                "mean_difference": -0.222,
                "sd": 0.025059928172283332,
                "ci95_low": -0.2842523126265235,
                "ci95_high": -0.15974768737347642,
                "exact_signflip_two_sided_p": 0.25,
            },
            "per_seed_differences": {
                "85000": -0.196,
                "85001": -0.224,
                "85002": -0.246,
            },
        }
        (summary_dir / "analysis_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report), "--data-root", str(data)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_accepts_hand_checked_audited_row(tmp_path):
    result = run_check(tmp_path, TABLE)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REPORT_TABLE_VALID 1"


def test_rejects_wrong_mean_even_when_other_numbers_match(tmp_path):
    result = run_check(tmp_path, TABLE.replace("-0,2220", "-0,1220"))
    assert result.returncode != 0
    assert "mean_difference" in result.stderr


def test_rejects_missing_audited_set(tmp_path):
    row = "| 2/2 | ccmplanner, V100 | -0,196 / -0,224 / -0,246 | -0,2220 | 0,0251 | от -0,2843 до -0,1597 | 0,25 |\n"
    result = run_check(tmp_path, TABLE.replace(row, ""))
    assert result.returncode != 0
    assert "missing" in result.stderr
