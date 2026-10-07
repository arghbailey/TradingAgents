"""Model roles, real/fake LLM construction, and the news and sentiment provider interfaces.

Every role maps to a (provider, model, base_url) triple in config, never in code.
Real clients come from TradingAgents' own ``create_llm_client`` factory, so any
provider upstream supports works here: openai, anthropic, google, openrouter,
ollama, openai_compatible, and the rest. ``offline`` builds scripted fakes that need
no key and no network.

LLM output is narrative only. Nothing an LLM returns can change a risk verdict;
see ``apex.risk``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage

ROLES = ("technical", "macro_news", "sentiment", "bull", "bear",
         "research_manager", "risk_narrative", "portfolio_manager")


@dataclass(frozen=True)
class RoleModel:
    provider: str          # any TradingAgents provider, or "stub" (no LLM; interface only)
    model: str
    base_url: str | None = None


# Local profile: everything on the user's own Ollama server, with the models
# run_aapl.py already uses. No key needed.
LOCAL_PROFILE: dict[str, RoleModel] = {
    "technical": RoleModel("ollama", "qwen2.5-coder:14b"),
    "macro_news": RoleModel("ollama", "qwen2.5-coder:7b"),
    "sentiment": RoleModel("ollama", "qwen2.5-coder:7b"),
    "bull": RoleModel("ollama", "qwen2.5-coder:7b"),
    "bear": RoleModel("ollama", "qwen2.5-coder:7b"),
    "research_manager": RoleModel("ollama", "qwen2.5-coder:14b"),
    "risk_narrative": RoleModel("ollama", "qwen2.5-coder:7b"),
    "portfolio_manager": RoleModel("ollama", "qwen2.5-coder:14b"),
}

# Cloud profile. Anthropic and OpenAI model IDs come from upstream's model catalog
# (tradingagents/llm_clients/model_catalog.py). OpenRouter and Perplexity IDs are
# not in that catalog; confirm them on openrouter.ai/models and docs.perplexity.ai.
CLOUD_PROFILE: dict[str, RoleModel] = {
    "technical": RoleModel("anthropic", "claude-sonnet-5"),
    # Perplexity speaks the OpenAI Chat Completions API; upstream has no
    # "perplexity" provider, so it goes through openai_compatible (key in
    # OPENAI_COMPATIBLE_API_KEY).
    "macro_news": RoleModel("openai_compatible", "sonar-pro", "https://api.perplexity.ai"),
    # "TypeSafe Jev" has no API we could find; sentiment is a stub interface.
    "sentiment": RoleModel("stub", "jev"),
    "bull": RoleModel("openrouter", "nousresearch/hermes-4-405b"),
    "bear": RoleModel("openrouter", "qwen/qwen3-235b-a22b"),
    "research_manager": RoleModel("anthropic", "claude-opus-5-5"),
    "risk_narrative": RoleModel("anthropic", "claude-sonnet-5"),
    "portfolio_manager": RoleModel("openai", "gpt-6-sol"),
}

PROFILES = {"local": LOCAL_PROFILE, "cloud": CLOUD_PROFILE}


def resolve_roles(profile: str = "local", overrides: dict[str, dict] | None = None,
                  environ: dict[str, str] | None = None) -> dict[str, RoleModel]:
    """Role map: the profile, then APEX_ROLE_<ROLE>_{PROVIDER,MODEL,BASE_URL} env vars, then ``overrides``."""
    if profile not in PROFILES:
        raise ValueError(f"unknown LLM profile {profile!r}; choose {', '.join(PROFILES)}")
    env = os.environ if environ is None else environ
    roles = dict(PROFILES[profile])
    for role in ROLES:
        cur = roles[role]
        prefix = f"APEX_ROLE_{role.upper()}_"
        roles[role] = RoleModel(
            env.get(prefix + "PROVIDER") or cur.provider,
            env.get(prefix + "MODEL") or cur.model,
            env.get(prefix + "BASE_URL") or cur.base_url,
        )
    for role, o in (overrides or {}).items():
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; roles: {', '.join(ROLES)}")
        cur = roles[role]
        roles[role] = RoleModel(o.get("provider", cur.provider), o.get("model", cur.model),
                                o.get("base_url", cur.base_url))
    return roles


# ---------------------------------------------------------------------- fake LLMs


class ScriptedLLM:
    """A deterministic stand-in for a chat model: ``invoke(prompt) -> AIMessage``."""

    def __init__(self, role: str, reply: str | None = None):
        self.role = role
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt: Any, *args, **kwargs) -> AIMessage:
        self.calls.append(str(prompt))
        return AIMessage(content=self.reply if self.reply is not None else _DEFAULT_REPLIES[self.role])


_DEFAULT_REPLIES = {
    "technical": "Technicals (offline fixture): trend intact above VWAP; pullback into the 21/30 EMA zone.",
    "macro_news": "Macro (offline fixture): CPI printed at 08:30 ET in line; FOMC statement at 14:00 ET.",
    "sentiment": "SCORE: 0.20",
    "bull": "Bull (offline fixture): trend, VWAP and EMA stack aligned; pullback entry offers defined risk.",
    "bear": "Bear (offline fixture): FOMC this afternoon; RSI elevated; size conservatively.",
    "research_manager": "Bull case carries on structure; risk is defined by the 30 EMA.\nRATING: BUY",
    "risk_narrative": "Risk (offline fixture): the governor's verdict stands; narrative only.",
    "portfolio_manager": "PM (offline fixture): execute the approved bracket as sized by the governor.",
}


def offline_llms(replies: dict[str, str] | None = None) -> dict[str, ScriptedLLM]:
    replies = replies or {}
    return {role: ScriptedLLM(role, replies.get(role)) for role in ROLES}


def build_role_llms(roles: dict[str, RoleModel], extra_kwargs: dict | None = None) -> dict[str, Any]:
    """Real chat models via TradingAgents' factory. Stub roles map to None."""
    from tradingagents.llm_clients import create_llm_client

    out: dict[str, Any] = {}
    for role, rm in roles.items():
        if rm.provider == "stub":
            out[role] = None
            continue
        client = create_llm_client(rm.provider, rm.model, rm.base_url, **(extra_kwargs or {}))
        out[role] = client.get_llm()
    return out


def text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


RATINGS = ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")
_RATING_RE = re.compile(r"RATING\s*[:=]\s*\**\s*(STRONG[_ ]BUY|STRONG[_ ]SELL|BUY|SELL|NEUTRAL)", re.I)


def parse_rating(text: str) -> str:
    """Strictly parse ``RATING: <x>``; anything unparseable is NEUTRAL (no trade)."""
    m = _RATING_RE.search(text or "")
    return m.group(1).upper().replace(" ", "_") if m else "NEUTRAL"


# ------------------------------------------------------- news / sentiment interfaces


@runtime_checkable
class NewsProvider(Protocol):
    def headlines(self, symbol: str, trade_date: str) -> list[str]: ...


@runtime_checkable
class SentimentProvider(Protocol):
    def score(self, symbol: str, trade_date: str) -> float: ...


class FixtureNews:
    def headlines(self, symbol: str, trade_date: str) -> list[str]:
        return ["CPI in line with consensus", "Fed speakers reiterate data dependence",
                "Mega-cap tech leads pre-market"]


class FixtureSentiment:
    def __init__(self, value: float = 0.2):
        self.value = value

    def score(self, symbol: str, trade_date: str) -> float:
        return self.value


class JevSentimentStub:
    """Placeholder for the "TypeSafe Jev" sentiment API named in the design docs.

    We found no public API by that name, so this raises until someone wires a real one.
    """

    def score(self, symbol: str, trade_date: str) -> float:
        raise NotImplementedError("TypeSafe Jev sentiment API is not available; "
                                  "inject a SentimentProvider or use LLMSentiment")


class LLMSentiment:
    """Asks the sentiment-role LLM for ``SCORE: x``; the result is clamped to [-1, 1]."""

    def __init__(self, llm: Any, news: NewsProvider):
        self.llm, self.news = llm, news

    def score(self, symbol: str, trade_date: str) -> float:
        heads = self.news.headlines(symbol, trade_date)
        reply = text_of(self.llm.invoke(
            f"Rate the sentiment of these headlines for {symbol} futures on {trade_date} "
            f"from -1 (bearish) to 1 (bullish). Answer 'SCORE: <number>'.\n- " + "\n- ".join(heads)))
        return parse_score(reply)


def parse_score(text: str) -> float:
    m = re.search(r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)", text or "", re.I)
    return clamp_sentiment(float(m.group(1))) if m else 0.0


def clamp_sentiment(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))
