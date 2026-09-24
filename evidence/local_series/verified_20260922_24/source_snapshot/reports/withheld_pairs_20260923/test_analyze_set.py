import csv
import io

import pytest

from analyze_set import csv_bytes, write_once


def test_analysis_table_has_stable_columns_and_rows():
    rows = [{"seed": 85000, "hit32": 0.5}, {"seed": 85001, "hit32": 0.4}]
    parsed = list(csv.DictReader(io.StringIO(csv_bytes(rows).decode("utf-8"))))
    assert parsed == [{"seed": "85000", "hit32": "0.5"},
                      {"seed": "85001", "hit32": "0.4"}]
    with pytest.raises(ValueError, match="empty"):
        csv_bytes([])


def test_analysis_output_is_idempotent_but_never_replaced(tmp_path):
    path = tmp_path / "audit" / "summary.json"
    write_once(path, b"first")
    write_once(path, b"first")
    with pytest.raises(ValueError, match="differs"):
        write_once(path, b"changed")
    assert path.read_bytes() == b"first"
