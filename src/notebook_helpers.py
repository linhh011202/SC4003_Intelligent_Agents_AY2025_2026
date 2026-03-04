from __future__ import annotations

import json
import subprocess
import sys
import base64
from html import escape
from pathlib import Path


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "main.py").exists():
        return cwd
    if (cwd.parent / "main.py").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate project root containing main.py")


def run_part(part: str) -> None:
    root = project_root()
    subprocess.run(
        [sys.executable, str(root / "main.py"), "--part", part],
        check=True,
        cwd=root,
    )


def load_json(path: str | Path) -> dict:
    target = project_root() / Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def svg_html(path: str | Path, title: str | None = None, width: str = "100%") -> str:
    target = project_root() / Path(path)
    svg = target.read_text(encoding="utf-8")
    title_html = f"<h4 style='margin:0 0 8px 0'>{escape(title)}</h4>" if title else ""
    return (
        "<div style='flex:1; min-width:320px; border:1px solid #ddd; border-radius:8px; "
        "padding:12px; background:#fff;'>"
        f"{title_html}"
        f"<div style='width:{escape(width)}'>{svg}</div>"
        "</div>"
    )


def image_html(path: str | Path, title: str | None = None, width: str = "100%") -> str:
    target = project_root() / Path(path)
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    suffix = target.suffix.lower().lstrip(".")
    title_html = f"<h4 style='margin:0 0 8px 0'>{escape(title)}</h4>" if title else ""
    return (
        "<div style='flex:1; min-width:320px; border:1px solid #ddd; border-radius:8px; "
        "padding:12px; background:#fff;'>"
        f"{title_html}"
        f"<img src='data:image/{suffix};base64,{encoded}' style='width:{escape(width)}; display:block;' />"
        "</div>"
    )


def row_html(*items: str) -> str:
    return (
        "<div style='display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start;'>"
        + "".join(items)
        + "</div>"
    )


def dict_table_html(mapping: dict, title: str | None = None) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr>"
            f"<td style='padding:8px; border:1px solid #ddd; font-family:monospace;'>{escape(str(key))}</td>"
            f"<td style='padding:8px; border:1px solid #ddd; font-family:monospace;'>{escape(str(value))}</td>"
            "</tr>"
        )
    title_html = f"<h4 style='margin:0 0 8px 0'>{escape(title)}</h4>" if title else ""
    return (
        f"{title_html}<table style='border-collapse:collapse; min-width:420px;'>"
        "<thead><tr><th style='padding:8px; border:1px solid #ddd;'>Metric</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def part2_summary_table(summary: dict) -> str:
    headers = [
        "Size",
        "VI Iter",
        "VI Runtime (ms)",
        "VI Mean Utility",
        "PI Iter",
        "PI Runtime (ms)",
        "PI Mean Utility",
        "Policy Mismatches",
    ]
    rows = []
    for row in summary["rows"]:
        values = [
            row["maze_size"],
            row["value_iteration_iterations"],
            row["value_iteration_runtime_ms"],
            row["value_iteration_mean_utility"],
            row["policy_iteration_iterations"],
            row["policy_iteration_runtime_ms"],
            row["policy_iteration_mean_utility"],
            row["policy_mismatches"],
        ]
        rows.append(
            "<tr>"
            + "".join(
                f"<td style='padding:8px; border:1px solid #ddd; font-family:monospace;'>{escape(str(value))}</td>"
                for value in values
            )
            + "</tr>"
        )
    header_html = "".join(
        f"<th style='padding:8px; border:1px solid #ddd; background:#f4f4f4;'>{escape(item)}</th>"
        for item in headers
    )
    return (
        "<table style='border-collapse:collapse; min-width:900px;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
