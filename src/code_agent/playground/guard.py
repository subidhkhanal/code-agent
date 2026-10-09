"""Spending and abuse limits for the public playground, enforced before any model call.

* A global daily budget in USD. Each run *reserves* a worst-case amount before it starts and
  settles to its measured cost when it ends, so concurrent runs can't overshoot the day.
* A per-visitor number of runs per day. A run counts when it starts, whether or not it succeeds.
* A cap on concurrent runs.

Visitors are identified by a salted hash of their IP; raw addresses are never stored. The day
rolls over at 00:00 UTC. Spend and run counts persist to a small JSON file so a process restart
doesn't reset the budget (the provider-side spend limit is the final backstop either way).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class RunRefusedError(RuntimeError):
    """The run can't start; the message is shown to the visitor as is."""


@dataclass(frozen=True)
class Ticket:
    visitor: str
    reserved_usd: float


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class SpendGuard:
    def __init__(
        self,
        state_file: Path,
        *,
        daily_usd: float,
        runs_per_visitor: int,
        max_concurrent: int,
        reserve_usd: float,
        clock=_today,
    ) -> None:
        self.state_file = state_file
        self.daily_usd = daily_usd
        self.runs_per_visitor = runs_per_visitor
        self.max_concurrent = max_concurrent
        self.reserve_usd = reserve_usd
        self._clock = clock
        self._lock = threading.Lock()
        self._salt = secrets.token_bytes(16)
        self._active = 0
        self._reserved = 0.0
        self._day, self._spent, self._runs = self._load()

    # -- identity -----------------------------------------------------------------------------

    def visitor_id(self, address: str) -> str:
        return hashlib.sha256(self._salt + address.encode()).hexdigest()[:16]

    # -- persistence --------------------------------------------------------------------------

    def _load(self) -> tuple[str, float, dict[str, int]]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if data.get("day") == self._clock():
                return data["day"], float(data["spent_usd"]), dict(data["runs"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return self._clock(), 0.0, {}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        payload = {"day": self._day, "spent_usd": round(self._spent, 6), "runs": self._runs}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.state_file)

    def _roll(self) -> None:
        if self._day != self._clock():
            self._day, self._spent, self._runs = self._clock(), 0.0, {}

    # -- API ----------------------------------------------------------------------------------

    def status(self, visitor: str) -> dict:
        with self._lock:
            self._roll()
            remaining = max(0.0, self.daily_usd - self._spent - self._reserved)
            return {
                "runs_left": max(0, self.runs_per_visitor - self._runs.get(visitor, 0)),
                "runs_per_day": self.runs_per_visitor,
                "budget_open": remaining >= self.reserve_usd,
                "busy": self._active >= self.max_concurrent,
            }

    def admit(self, visitor: str) -> Ticket:
        with self._lock:
            self._roll()
            if self._runs.get(visitor, 0) >= self.runs_per_visitor:
                raise RunRefusedError(
                    f"You've used your {self.runs_per_visitor} runs for today. "
                    "Come back tomorrow (UTC)."
                )
            if self._spent + self._reserved + self.reserve_usd > self.daily_usd:
                raise RunRefusedError(
                    "Today's playground budget is used up. It resets at 00:00 UTC."
                )
            if self._active >= self.max_concurrent:
                raise RunRefusedError("The agent is busy with other visitors. Try again shortly.")
            self._runs[visitor] = self._runs.get(visitor, 0) + 1
            self._reserved += self.reserve_usd
            self._active += 1
            self._save()
            return Ticket(visitor, self.reserve_usd)

    def settle(self, ticket: Ticket, cost_usd: float | None) -> None:
        """Release the reservation and record the real cost. An unknown cost is charged as the
        full reservation: unknown spend must never look like free spend."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - ticket.reserved_usd)
            self._active = max(0, self._active - 1)
            self._roll()
            self._spent += ticket.reserved_usd if cost_usd is None else cost_usd
            self._save()
