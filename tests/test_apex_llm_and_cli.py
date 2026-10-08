"""LLM role configuration and the run_apex_automation CLI. Offline, no keys."""

from __future__ import annotations

import json

import pytest

import run_apex_automation as cli
from apex.llm import (
    LOCAL_PROFILE,
    build_role_llms,
    parse_rating,
    parse_score,
    resolve_roles,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("text,rating", [
    ("RATING: BUY", "BUY"), ("rating = strong_sell", "STRONG_SELL"), ("**RATING**: ...", "NEUTRAL"),
    ("RATING: Strong Buy", "STRONG_BUY"), ("buy buy buy", "NEUTRAL"), ("", "NEUTRAL"),
])
def test_parse_rating(text, rating):
    assert parse_rating(text) == rating


def test_parse_score_clamps():
    assert parse_score("SCORE: 3.5") == 1.0
    assert parse_score("SCORE: -0.4") == -0.4
    assert parse_score("no idea") == 0.0


def test_roles_default_local_and_env_override():
    roles = resolve_roles("local", environ={})
    assert roles == LOCAL_PROFILE
    assert all(r.provider == "ollama" for r in roles.values())
    roles = resolve_roles("cloud", environ={"APEX_ROLE_BULL_MODEL": "x/y", "APEX_ROLE_BEAR_PROVIDER": "ollama"})
    assert roles["bull"].provider == "openrouter" and roles["bull"].model == "x/y"
    assert roles["bear"].provider == "ollama"
    assert roles["sentiment"].provider == "stub"
    with pytest.raises(ValueError):
        resolve_roles("nope", environ={})
    with pytest.raises(ValueError):
        resolve_roles("local", overrides={"astrologer": {}}, environ={})


def test_build_role_llms_uses_upstream_factory_without_network():
    llms = build_role_llms(resolve_roles("local", environ={}))
    assert type(llms["bull"]).__name__ == "LocalCompatibleChatOpenAI"
    llms = build_role_llms({"sentiment": resolve_roles("cloud", environ={})["sentiment"]})
    assert llms["sentiment"] is None


@pytest.mark.parametrize("mode", ["live", "paper", "LIVE", "dry_run"])
def test_cli_rejects_any_mode_but_dry_run(mode, capsys):
    assert cli.main(["--mode", mode, "--offline"]) == 2
    assert "DRY-RUN ONLY" in capsys.readouterr().err


def test_cli_requires_bars_when_not_offline(tmp_path, capsys):
    assert cli.main(["--mode", "dry-run", "--vault", str(tmp_path / "v")]) == 2
    assert "--bars-csv" in capsys.readouterr().err


def test_cli_offline_end_to_end(tmp_path, capsys):
    rc = cli.main(["--mode", "dry-run", "--tier", "50K", "--eval-type", "EOD", "--offline",
                   "--date", "2026-10-07", "--vault", str(tmp_path / "v"), "--out", str(tmp_path / "o")])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"risk_verdict": "APPROVED"' in out and "not sent to any broker" in out
    assert len(list((tmp_path / "o").glob("*.json"))) == 1
    assert (tmp_path / "v" / "raw" / "executions" / "2026-10-07.json").exists()


def test_cli_account_json_drives_halt(tmp_path, capsys):
    acct = tmp_path / "acct.json"
    acct.write_text(json.dumps({"balance": 49_000.0}), encoding="utf-8")  # -1,000 on the day
    rc = cli.main(["--mode", "dry-run", "--offline", "--date", "2026-10-07", "--account-json", str(acct),
                   "--vault", str(tmp_path / "v"), "--out", str(tmp_path / "o")])
    assert rc == 0
    assert '"order_action": "HALT_FLATTEN"' in capsys.readouterr().out


def test_cli_bad_tier(capsys):
    assert cli.main(["--mode", "dry-run", "--offline", "--tier", "75K"]) == 2
