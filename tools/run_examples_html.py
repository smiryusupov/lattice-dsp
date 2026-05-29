from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; line-height: 1.45; }
main { max-width: 1200px; margin: 0 auto; }
.summary { display: flex; gap: 1rem; flex-wrap: wrap; }
.card, details { border: 1px solid #9995; border-radius: 12px; padding: 1rem; margin: 1rem 0; }
.ok { color: #16833a; font-weight: 700; }
.fail { color: #b00020; font-weight: 700; }
pre { overflow-x: auto; padding: 1rem; border-radius: 10px; background: color-mix(in srgb, CanvasText 8%, Canvas); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.small { color: color-mix(in srgb, CanvasText 65%, Canvas); font-size: 0.9rem; }
summary { cursor: pointer; font-size: 1.05rem; }
"""


def run_example(
    path: Path, *, timeout: float, env: dict[str, str], artifact_dir: Path
) -> dict[str, object]:
    start = time.perf_counter()

    try:
        local_env = env.copy()
        local_env["LATTICE_DSP_ARTIFACT_DIR"] = str(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=Path.cwd(),
            env=local_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return {
            "path": str(path),
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "status": "ok" if proc.returncode == 0 else "fail",
            "artifact_dir": str(artifact_dir),
        }

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "path": str(path),
            "returncode": None,
            "elapsed_s": elapsed,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimed out after {timeout} seconds.",
            "status": "fail",
            "artifact_dir": str(artifact_dir),
        }


def render(results: list[dict[str, object]], *, title: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "fail")

    parts = [
        "<!doctype html>",
        "<html lang='en'><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(title)}</title>",
        f"<style>{CSS}</style>",
        "</head><body><main>",
        f"<h1>{escape(title)}</h1>",
        f"<p class='small'>Generated {escape(now)} with <code>{escape(sys.executable)}</code>.</p>",
        "<section class='summary'>",
        f"<div class='card'><strong>Total</strong><br>{len(results)}</div>",
        f"<div class='card'><strong>Passed</strong><br><span class='ok'>{ok}</span></div>",
        f"<div class='card'><strong>Failed</strong><br><span class='fail'>{fail}</span></div>",
        "</section>",
    ]

    for r in results:
        status = str(r["status"])
        klass = "ok" if status == "ok" else "fail"
        path = str(r["path"])
        elapsed = float(r["elapsed_s"])
        returncode = r["returncode"]
        open_attr = "" if status == "ok" else " open"

        parts.append(f"<details{open_attr}>")
        parts.append(
            f"<summary><span class='{klass}'>{escape(status.upper())}</span> "
            f"<code>{escape(path)}</code> "
            f"<span class='small'>({elapsed:.3f}s, returncode={escape(str(returncode))})</span></summary>"
        )
        stdout = str(r.get("stdout", "")).strip()
        stderr = str(r.get("stderr", "")).strip()
        if stdout:
            parts.append("<h3>stdout</h3>")
            parts.append("<pre>" + escape(stdout) + "</pre>")
        if stderr:
            parts.append("<h3>stderr</h3>")
            parts.append("<pre>" + escape(stderr) + "</pre>")
        artifact_dir = Path(str(r.get("artifact_dir", "")))
        if artifact_dir.exists():
            files = sorted(p for p in artifact_dir.iterdir() if p.is_file())
            if files:
                parts.append("<h3>artifacts</h3><ul>")
                for file in files:
                    parts.append(f"<li><code>{escape(str(file))}</code></li>")
                parts.append("</ul>")
        parts.append("</details>")

    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run examples and write a static HTML report of stdout/stderr."
    )
    parser.add_argument("--examples-dir", type=Path, default=Path("examples"))
    parser.add_argument("--output", type=Path, default=Path("reports/examples.html"))
    parser.add_argument("--pattern", default="*.py")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--title", default="lattice-dsp examples report")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")

    examples = sorted(args.examples_dir.glob(args.pattern))
    if not examples:
        raise SystemExit(f"No examples matched {args.examples_dir / args.pattern}")

    results = []

    for path in examples:
        print(f"Running {path}...")
        artifact_dir = Path("reports/example-artifacts") / path.stem
        result = run_example(path, timeout=args.timeout, env=env, artifact_dir=artifact_dir)
        results.append(result)

        if args.fail_fast and result["status"] != "ok":
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results, title=args.title), encoding="utf-8")
    print(f"Wrote {args.output}")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"{len(failed)} example(s) failed. Open {args.output} for details.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
