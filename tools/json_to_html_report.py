from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; line-height: 1.45; }
main { max-width: 1200px; margin: 0 auto; }
h1, h2 { line-height: 1.15; }
.card { border: 1px solid #9995; border-radius: 12px; padding: 1rem; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
th, td { border-bottom: 1px solid #9995; padding: 0.45rem 0.55rem; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { position: sticky; top: 0; background: Canvas; }
tr:hover { background: color-mix(in srgb, CanvasText 7%, Canvas); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { overflow-x: auto; padding: 1rem; border-radius: 10px; background: color-mix(in srgb, CanvasText 8%, Canvas); }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }
.meta-item { border: 1px solid #9994; border-radius: 10px; padding: 0.6rem; }
.small { color: color-mix(in srgb, CanvasText 65%, Canvas); font-size: 0.9rem; }
"""


def fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return escape(str(value))
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return escape(str(value))
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 1e-3:
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return escape(", ".join(str(v) for v in value))
    if isinstance(value, dict):
        return "<pre>" + escape(json.dumps(value, indent=2, sort_keys=True)) + "</pre>"
    return escape(str(value))


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    else:
        out[prefix] = value


def table_html(rows: list[dict[str, Any]], *, first_col: str | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    if first_col and first_col in columns:
        columns = [first_col] + [c for c in columns if c != first_col]

    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{escape(col)}</th>" for col in columns)
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        for col in columns:
            parts.append(f"<td>{fmt(row.get(col, ''))}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "\n".join(parts)


def benchmark_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, values in payload.get("benchmarks", {}).items():
        row = {"benchmark": name}
        if isinstance(values, dict):
            row.update(values)
        else:
            row["value"] = values
        rows.append(row)

    return sorted(
        rows,
        key=lambda r: float(r["median_s"]) if isinstance(r.get("median_s"), int | float) else 1e99,
    )


def payload_rows(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    if isinstance(payload.get("benchmarks"), dict):
        return "Benchmark timings", benchmark_rows(payload), "benchmark"

    if isinstance(payload.get("results"), list):
        return "Results", [dict(r) for r in payload["results"] if isinstance(r, dict)], None

    if isinstance(payload.get("rows"), list):
        return "Rows", [dict(r) for r in payload["rows"] if isinstance(r, dict)], None

    if isinstance(payload.get("scenarios"), list):
        rows = []
        for idx, scenario in enumerate(payload["scenarios"], start=1):
            flat: dict[str, Any] = {"scenario_index": idx}
            flatten("", scenario, flat)
            rows.append(flat)
        return "Scenarios", rows, "scenario_index"

    flat = []
    for key, value in payload.items():
        if key != "metadata":
            flat.append({"key": key, "value": value})
    return "Payload", flat, "key"


def render(payload: dict[str, Any], *, title: str, source: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata = payload.get("metadata", {})
    section_title, rows, first_col = payload_rows(payload)

    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(title)}</title>",
        f"<style>{CSS}</style>",
        "</head><body><main>",
        f"<h1>{escape(title)}</h1>",
        f"<p class='small'>Generated {escape(now)} from <code>{escape(str(source))}</code>.</p>",
    ]

    if isinstance(metadata, dict) and metadata:
        parts.append("<section class='card'><h2>Metadata</h2><div class='meta-grid'>")
        for key, value in metadata.items():
            parts.append(
                "<div class='meta-item'>"
                f"<div class='small'>{escape(str(key))}</div>"
                f"<div>{fmt(value)}</div>"
                "</div>"
            )
        parts.append("</div></section>")

    parts.append(f"<section class='card'><h2>{escape(section_title)}</h2>")
    parts.append(table_html(rows, first_col=first_col))
    parts.append("</section>")

    parts.append("<section class='card'><h2>Raw JSON</h2>")
    parts.append("<pre>" + escape(json.dumps(payload, indent=2, sort_keys=True)) + "</pre>")
    parts.append("</section>")

    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a lattice-dsp JSON result file to a static HTML report."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    title = args.title or args.input.stem.replace("-", " ").replace("_", " ").title()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload, title=title, source=args.input), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
