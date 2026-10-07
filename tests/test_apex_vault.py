"""Vault: raw/ is never overwritten, log.md is append-only, stats and index update."""

from __future__ import annotations

import pytest

from apex.vault import Vault

pytestmark = pytest.mark.unit


def _record(day="2026-10-07", verdict="APPROVED", setup="RMA"):
    return {"symbol": "MNQ", "trade_date": day, "active_setup": setup, "risk_verdict": verdict,
            "order_action": "BUY" if verdict == "APPROVED" else "NO_TRADE",
            "order_payload": {"symbol": "MNQ"} if verdict == "APPROVED" else None,
            "order_contracts": 1, "consensus_rating": "BUY", "eval_type": "EOD"}


def test_layout_created(tmp_path):
    v = Vault(tmp_path)
    v.ensure_layout()
    for d in ("raw/market_data", "raw/news", "raw/executions", "wiki/strategies", "wiki/entities",
              "wiki/accounts", "wiki/regimes", "wiki/post-mortems"):
        assert (tmp_path / d).is_dir()
    assert (tmp_path / "index.md").exists() and (tmp_path / "log.md").exists()


def test_write_raw_never_overwrites(tmp_path):
    v = Vault(tmp_path)
    first = v.write_raw("executions/2026-10-07.json", '{"n": 1}')
    second = v.write_raw("executions/2026-10-07.json", '{"n": 2}')
    third = v.write_raw("executions/2026-10-07.json", '{"n": 3}')
    assert first.name == "2026-10-07.json"
    assert second.name == "2026-10-07-2.json" and third.name == "2026-10-07-3.json"
    assert first.read_text() == '{"n": 1}'


def test_write_raw_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        Vault(tmp_path).write_raw("../wiki/evil.md", "x")


def test_reflect_twice_preserves_existing_raw_and_appends_log(tmp_path):
    v = Vault(tmp_path)
    r1 = v.reflect(_record())
    raw1 = (tmp_path / "raw" / "executions" / "2026-10-07.json")
    before = raw1.read_bytes()
    log_before = (tmp_path / "log.md").read_text(encoding="utf-8")
    r2 = v.reflect(_record(verdict="REJECT"))
    assert raw1.read_bytes() == before
    assert r1["raw_path"] != r2["raw_path"]
    log_after = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert log_after.startswith(log_before)
    assert log_after.count("## [2026-10-07] reflect | ") == 2
    pm = (tmp_path / "wiki" / "post-mortems" / "2026-10-07.md").read_text(encoding="utf-8")
    assert pm.count("## MNQ") == 2
    assert "[[wiki/post-mortems/2026-10-07]]" in (tmp_path / "index.md").read_text(encoding="utf-8")


def test_stats_win_rate_and_ev(tmp_path):
    v = Vault(tmp_path)
    v.ensure_layout()
    v.record_decision("RMA", approved=True)
    v.record_outcome("RMA", pnl=24.0, r_multiple=0.75)
    v.record_outcome("RMA", pnl=-32.0, r_multiple=-1.0)
    s = v.record_outcome("RMA", pnl=40.0, r_multiple=1.25)
    assert s["wins"] == 2 and s["losses"] == 1
    assert s["win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["ev_r"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["ev_pnl"] == pytest.approx(32 / 3, abs=0.01)
    note = (tmp_path / "wiki" / "strategies" / "RMA.md").read_text(encoding="utf-8")
    assert "66.7%" in note and note.startswith("# RMA")


def test_preflight_reads_entity_and_strategy_notes(tmp_path):
    v = Vault(tmp_path)
    v.ensure_layout()
    (tmp_path / "wiki" / "entities" / "MNQ.md").write_text("# MNQ\nThin at lunch.", encoding="utf-8")
    ctx = v.preflight("mnq", "50K")
    assert "Thin at lunch" in ctx["entity"]
    assert set(ctx["strategies"]) >= {"RMA", "FFMA", "TREND", "MOMO", "DB_DT"}
