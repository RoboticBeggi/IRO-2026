"""Doctor engine: a registry of checks, a runner and a printed report.

A check is a function that returns one Result or a list of them. Every result
says what is wrong (detail) and what to do about it (fix), so the report reads
as a diagnosis, not just a red light.
"""

import time
import traceback
from dataclasses import dataclass, field

OK, INFO, WARN, FAIL, SKIP = "OK", "INFO", "WARN", "FAIL", "SKIP"
SEVERITY = {OK: 0, INFO: 0, SKIP: 0, WARN: 1, FAIL: 2}


@dataclass
class Result:
    group: str
    name: str
    status: str
    detail: str
    fix: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Check:
    group: str
    name: str
    func: callable
    quick: bool   # cheap enough for `doctor watch` (runs every few seconds)
    slow: bool    # skipped by `doctor quick`


CHECKS: list[Check] = []


def check(group: str, name: str, quick: bool = False, slow: bool = False):
    def register(func):
        CHECKS.append(Check(group, name, func, quick, slow))
        return func
    return register


def run(groups: list[str] | None = None, mode: str = "full", cfg: dict | None = None) -> list[Result]:
    """mode: 'full' runs everything, 'quick' skips slow checks, 'watch' only the quick ones."""
    from iro import load_config
    cfg = cfg or load_config()
    results = []
    for c in CHECKS:
        if groups and c.group not in groups:
            continue
        if (mode == "quick" and c.slow) or (mode == "watch" and not c.quick):
            continue
        try:
            out = c.func(cfg)
        except Exception as e:  # noqa: BLE001 - a broken check must not stop the others
            out = Result(c.group, c.name, FAIL, f"check crashed: {type(e).__name__}: {e}",
                         "This is a bug in the doctor or an unexpected device reply. Details: "
                         + traceback.format_exc(limit=1).strip().splitlines()[-1])
        results.extend(out if isinstance(out, list) else [out])
    return results


def worst(results: list[Result]) -> int:
    return max((SEVERITY[r.status] for r in results), default=0)


_COLORS = {OK: "\033[32m", INFO: "\033[36m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_ICONS = {OK: "✔", INFO: "i", WARN: "⚠", FAIL: "✖", SKIP: "-"}


def print_report(results: list[Result], color: bool = True, verbose: bool = False) -> None:
    def c(status, text):
        return f"{_COLORS[status]}{text}\033[0m" if color else text

    group = None
    for r in results:
        if r.group != group:
            group = r.group
            print(f"\n\033[1m{group.upper()}\033[0m" if color else f"\n{group.upper()}")
        if r.status in (OK, SKIP, INFO) and not verbose:
            print(f"  {c(r.status, _ICONS[r.status])} {r.name}: {r.detail}")
            continue
        print(f"  {c(r.status, _ICONS[r.status] + ' ' + r.status)} {r.name}: {r.detail}")
        if r.fix:
            print(f"      → {r.fix}")

    counts = {s: sum(r.status == s for r in results) for s in (OK, WARN, FAIL)}
    verdict = {0: c(OK, "HEALTHY"), 1: c(WARN, "NEEDS ATTENTION"), 2: c(FAIL, "NOT SAFE TO RUN")}[worst(results)]
    print(f"\n{verdict}  ({counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} failures)"
          f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
