"""Print the public top-level lattice_dsp API for release review.

This script is intentionally simple.  It helps maintainers inspect exported
symbols before a release and catch accidental public names.  It does not decide
whether a symbol is stable; it just makes the surface visible.
"""

from __future__ import annotations

import argparse
import inspect
from collections import defaultdict
from types import ModuleType
from typing import Any

import lattice_dsp as ld


DEPRECATED_ALIASES = {
    "finite_hankel_aak_reduce_impulse": "use finite_hankel_reduce_impulse",
    "finite_hankel_aak_reduce_iir": "use finite_hankel_reduce_iir",
}


def classify(obj: Any) -> str:
    if inspect.isclass(obj):
        return "classes"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "functions"
    if isinstance(obj, ModuleType):
        return "modules"
    return "constants/other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print public lattice_dsp symbols grouped by kind."
    )
    parser.add_argument(
        "--include-modules", action="store_true", help="include imported submodules in output"
    )
    parser.add_argument(
        "--hide-deprecated", action="store_true", help="omit known deprecated compatibility aliases"
    )
    args = parser.parse_args()

    groups: dict[str, list[str]] = defaultdict(list)
    for name in sorted(n for n in dir(ld) if not n.startswith("_")):
        if args.hide_deprecated and name in DEPRECATED_ALIASES:
            continue
        obj = getattr(ld, name)
        kind = classify(obj)
        if kind == "modules" and not args.include_modules:
            continue
        groups[kind].append(name)

    for kind in ["classes", "functions", "constants/other", "modules"]:
        names = groups.get(kind, [])
        if not names:
            continue
        print(f"{kind} ({len(names)})")
        print("-" * (len(kind) + len(str(len(names))) + 3))
        for name in names:
            suffix = (
                f"  (deprecated alias; {DEPRECATED_ALIASES[name]})"
                if name in DEPRECATED_ALIASES
                else ""
            )
            print(f"{name}{suffix}")
        print()


if __name__ == "__main__":
    main()
