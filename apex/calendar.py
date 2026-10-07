"""Injectable high-impact economic calendar.

No live calendar feed is wired in (see docs/APEX_AUTOMATION.md, "Not yet built").
``StaticCalendar`` takes events you supply, for example from a JSON file. ``fixture_calendar``
is the offline default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EconomicEvent:
    name: str          # e.g. "CPI", "FOMC"
    when: datetime     # timezone-aware
    impact: str = "high"

    def to_dict(self) -> dict:
        return {"name": self.name, "when": self.when.isoformat(), "impact": self.impact}


@runtime_checkable
class EconomicCalendar(Protocol):
    def events_for(self, day: date) -> list[EconomicEvent]: ...


class StaticCalendar:
    """A calendar that serves a fixed list of events."""

    def __init__(self, events: list[EconomicEvent] | None = None):
        self._events = list(events or [])

    def events_for(self, day: date) -> list[EconomicEvent]:
        return [e for e in self._events if e.when.astimezone(ET).date() == day]

    @classmethod
    def from_json(cls, path: str | Path) -> StaticCalendar:
        """Load ``[{"name": "CPI", "when": "2026-10-07T08:30:00-04:00", "impact": "high"}, ...]``."""
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        events = []
        for row in rows:
            when = datetime.fromisoformat(row["when"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=ET)
            events.append(EconomicEvent(row["name"], when, row.get("impact", "high")))
        return cls(events)


def at_et(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=ET)


def fixture_calendar(day: date) -> StaticCalendar:
    """Offline fixture: a CPI print at 08:30 ET and an FOMC statement at 14:00 ET on ``day``."""
    return StaticCalendar([
        EconomicEvent("CPI", at_et(day, 8, 30)),
        EconomicEvent("FOMC", at_et(day, 14, 0)),
    ])
