"""Obsidian / Karpathy-wiki style memory vault.

Layout (root is configurable; default ./vault)::

    raw/                       immutable source records, never modified once written
      market_data/
      news/
      executions/YYYY-MM-DD.json   (a second run that day writes YYYY-MM-DD-2.json, ...)
    wiki/                      curated, editable knowledge
      strategies/  entities/  accounts/  regimes/  post-mortems/YYYY-MM-DD.md
    index.md                   regenerated catalogue of the wiki
    log.md                     append-only: "## [YYYY-MM-DD] <op> | <summary>"

Raw files are created with exclusive-create mode ("x"), so an existing raw file
cannot be overwritten even by mistake.
"""

from __future__ import annotations

import json
from pathlib import Path

from .setups import SETUPS

_STATS_BEGIN = "<!-- apex-stats:begin -->"
_STATS_END = "<!-- apex-stats:end -->"

RAW_DIRS = ("raw/market_data", "raw/news", "raw/executions")
WIKI_DIRS = ("wiki/strategies", "wiki/entities", "wiki/accounts", "wiki/regimes", "wiki/post-mortems")

_STRATEGY_NOTES = {
    "RMA": "Pullback in trend to the 21/30 EMA zone; enter on the 21 EMA retest.",
    "FFMA": "Exhaustion fade when RSI > 80 (short) or < 20 (long).",
    "TREND": "Compressed 9/15 EMA ribbon crossover; never chase more than 3 bars past the cross.",
    "MOMO": "High-RVOL momentum bar in the direction of a fresh 9/15 crossover.",
    "DB_DT": "Double bottom / top with at least 10 pts (ES) / 30 pts (NQ) of height.",
}


class Vault:
    def __init__(self, root: str | Path = "vault"):
        self.root = Path(root)

    # ------------------------------------------------------------------ layout
    def ensure_layout(self) -> None:
        for d in RAW_DIRS + WIKI_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        log = self.root / "log.md"
        if not log.exists():
            log.write_text("# Vault log\n\nAppend-only.\n\n", encoding="utf-8")
        for name, note in _STRATEGY_NOTES.items():
            path = self.root / "wiki" / "strategies" / f"{name}.md"
            if not path.exists():
                path.write_text(f"# {name}\n\n{note}\n\n{_STATS_BEGIN}\n_No trades yet._\n{_STATS_END}\n",
                                encoding="utf-8")
        if not (self.root / "index.md").exists():
            self.update_index()

    # --------------------------------------------------------------------- raw
    def write_raw(self, relative: str, content: str) -> Path:
        """Create a new raw file. Never overwrites: a taken name gets a -2, -3, ... suffix."""
        target = self.root / "raw" / relative
        if (self.root / "raw").resolve() not in target.resolve().parents:
            raise ValueError(f"raw path escapes the vault: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        stem, suffix = target.stem, target.suffix
        n = 1
        while True:
            candidate = target if n == 1 else target.with_name(f"{stem}-{n}{suffix}")
            try:
                with candidate.open("x", encoding="utf-8") as fh:
                    fh.write(content)
                return candidate
            except FileExistsError:
                n += 1

    # --------------------------------------------------------------------- log
    def append_log(self, day: str, op: str, summary: str) -> None:
        with (self.root / "log.md").open("a", encoding="utf-8") as fh:
            fh.write(f"## [{day}] {op} | {summary}\n")

    # ---------------------------------------------------------------- preflight
    def _read(self, *parts: str) -> str:
        path = self.root.joinpath(*parts)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def preflight(self, symbol: str, tier: str) -> dict:
        self.ensure_layout()
        strategies = {p.stem: p.read_text(encoding="utf-8")
                      for p in sorted((self.root / "wiki" / "strategies").glob("*.md"))}
        return {
            "entity": self._read("wiki", "entities", f"{symbol.upper()}.md"),
            "account": self._read("wiki", "accounts", f"{tier.upper()}.md"),
            "strategies": strategies,
            "stats": self.load_stats(),
        }

    # -------------------------------------------------------------------- stats
    def _stats_path(self) -> Path:
        return self.root / "wiki" / "strategies" / "stats.json"

    def load_stats(self) -> dict:
        path = self._stats_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def _derive(s: dict) -> dict:
        closed = s.get("wins", 0) + s.get("losses", 0)
        s["win_rate"] = round(s["wins"] / closed, 4) if closed else None
        s["ev_r"] = round(s["total_r"] / closed, 4) if closed else None
        s["ev_pnl"] = round(s["total_pnl"] / closed, 2) if closed else None
        return s

    def _save_stats(self, stats: dict) -> None:
        self._stats_path().write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
        for setup, s in stats.items():
            self._render_stats(setup, s)

    def _blank(self) -> dict:
        return {"decisions": 0, "approved": 0, "wins": 0, "losses": 0, "total_r": 0.0, "total_pnl": 0.0}

    def record_decision(self, setup: str | None, approved: bool) -> dict:
        stats = self.load_stats()
        key = setup or "NONE"
        s = stats.setdefault(key, self._blank())
        s["decisions"] += 1
        s["approved"] += int(approved)
        stats[key] = self._derive(s)
        self._save_stats(stats)
        return stats[key]

    def record_outcome(self, setup: str, pnl: float, r_multiple: float) -> dict:
        """Record a closed trade's result (dry runs have none; call this when one exists)."""
        stats = self.load_stats()
        s = stats.setdefault(setup, self._blank())
        if pnl > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["total_r"] = round(s["total_r"] + r_multiple, 6)
        s["total_pnl"] = round(s["total_pnl"] + pnl, 2)
        stats[setup] = self._derive(s)
        self._save_stats(stats)
        return stats[setup]

    def _render_stats(self, setup: str, s: dict) -> None:
        if setup not in SETUPS:
            return
        path = self.root / "wiki" / "strategies" / f"{setup}.md"
        text = path.read_text(encoding="utf-8") if path.exists() else f"# {setup}\n\n{_STATS_BEGIN}\n{_STATS_END}\n"
        fmt = lambda v, pct=False: "n/a" if v is None else (f"{v:.1%}" if pct else f"{v}")  # noqa: E731
        block = (f"{_STATS_BEGIN}\n| decisions | approved | wins | losses | win rate | EV (R) | EV ($) |\n"
                 f"|---|---|---|---|---|---|---|\n"
                 f"| {s['decisions']} | {s['approved']} | {s['wins']} | {s['losses']} | "
                 f"{fmt(s['win_rate'], True)} | {fmt(s['ev_r'])} | {fmt(s['ev_pnl'])} |\n{_STATS_END}")
        if _STATS_BEGIN in text and _STATS_END in text:
            head, rest = text.split(_STATS_BEGIN, 1)
            _, tail = rest.split(_STATS_END, 1)
            text = head + block + tail
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        path.write_text(text, encoding="utf-8")

    # -------------------------------------------------------------------- index
    def update_index(self) -> None:
        lines = ["# Vault index", "", "Regenerated by the post-trade reflector.", ""]
        for section in ("strategies", "entities", "accounts", "regimes", "post-mortems"):
            pages = sorted((self.root / "wiki" / section).glob("*.md"))
            lines.append(f"## {section}")
            lines += [f"- [[wiki/{section}/{p.stem}]]" for p in pages] or ["- _(empty)_"]
            lines.append("")
        stats = self.load_stats()
        if stats:
            lines += ["## Setup stats", "", "| setup | decisions | approved | win rate | EV (R) |",
                      "|---|---|---|---|---|"]
            for k, s in sorted(stats.items()):
                wr = "n/a" if s.get("win_rate") is None else f"{s['win_rate']:.1%}"
                ev = "n/a" if s.get("ev_r") is None else f"{s['ev_r']}"
                lines.append(f"| {k} | {s['decisions']} | {s['approved']} | {wr} | {ev} |")
            lines.append("")
        (self.root / "index.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------------- reflection
    def write_post_mortem(self, day: str, markdown: str) -> Path:
        path = self.root / "wiki" / "post-mortems" / f"{day}.md"
        prefix = "" if path.exists() else f"# Post-mortem {day}\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(prefix + "\n" + markdown.rstrip() + "\n")
        return path

    def reflect(self, record: dict) -> dict:
        """Post-trade reflection: raw execution, post-mortem, stats, log, index."""
        self.ensure_layout()
        day = record["trade_date"]
        raw_path = self.write_raw(f"executions/{day}.json", json.dumps(record, indent=2, default=str))
        approved = record.get("risk_verdict") == "APPROVED" and record.get("order_payload") is not None
        stats = self.record_decision(record.get("active_setup"), approved)
        md = [
            f"## {record.get('symbol')} - {record.get('order_action')}",
            f"- setup: {record.get('active_setup')} | rating: {record.get('consensus_rating')} "
            f"| verdict: {record.get('risk_verdict')}",
            f"- contracts: {record.get('order_contracts')} | eval: {record.get('eval_type')}",
            f"- reason: {record.get('terminal_reason') or 'dry-run order written'}",
            f"- risk reasons: {'; '.join(record.get('risk_reasons') or []) or 'n/a'}",
            f"- raw record: `{raw_path.relative_to(self.root).as_posix()}`",
        ]
        pm_path = self.write_post_mortem(day, "\n".join(md))
        self.append_log(day, "reflect", f"{record.get('symbol')} {record.get('order_action')} "
                                        f"({record.get('active_setup')}, {record.get('risk_verdict')})")
        self.update_index()
        return {"raw_path": str(raw_path), "post_mortem": str(pm_path), "stats": stats}
