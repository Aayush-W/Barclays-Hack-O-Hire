from __future__ import annotations

import json
from pathlib import Path

from core.pattern_parser import parse_patterns_file


def test_parse_patterns_preserves_hyphenated_typology(tmp_path: Path) -> None:
    patterns_path = tmp_path / "patterns.txt"
    output_dir = tmp_path / "cases"
    patterns_path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 3-degree Fan-In",
                "2022/09/01 05:14,001,A1,002,B1,1000,USD,1000,USD,ACH",
                "END LAUNDERING ATTEMPT - FAN-IN",
                "BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 2-degree Fan-In",
                "2022/09/01 05:20,003,C1,004,D1,2000,USD,2000,USD,WIRE",
                "END LAUNDERING ATTEMPT - GATHER-SCATTER",
            ]
        ),
        encoding="utf-8",
    )

    parse_patterns_file(str(patterns_path), output_folder=str(output_dir))

    case_1 = json.loads((output_dir / "CASE_001.json").read_text(encoding="utf-8"))
    case_2 = json.loads((output_dir / "CASE_002.json").read_text(encoding="utf-8"))
    assert case_1["typology"] == "FAN-IN"
    assert case_2["typology"] == "GATHER-SCATTER"
