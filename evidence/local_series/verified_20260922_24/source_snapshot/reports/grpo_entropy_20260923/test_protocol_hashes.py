import re
from pathlib import Path


def test_frozen_input_hashes_are_full_sha256_values():
    protocol = Path(__file__).with_name("PROTOCOL.md").read_text(encoding="utf-8")
    for line in protocol.splitlines():
        if "SHA-256" in line or line.strip().startswith("`CD2A"):
            for value in re.findall(r"`([A-Fa-f0-9]{32,64})`", line):
                assert len(value) == 64
    assert "CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA" in protocol
