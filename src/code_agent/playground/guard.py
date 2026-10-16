"""Spending and abuse limits for the public playground, enforced before any model call.

Two kinds of run:

* **Free runs** use the owner's API key. Each device gets a fixed number, ever (not per day).
  A device is a random id the browser keeps; because clearing browser storage makes a new one,
  each network address also has a cap. Free runs draw on a global daily budget in USD: each run
  *reserves* a worst-case amount before it starts and settles to its measured cost when it ends,
  so concurrent runs can't overshoot the day.
* **Bring-your-own-key runs** use the visitor's key, so they skip the free quota and the daily
  budget. They still count toward the concurrency cap, which protects the shared CPU.

Devices and addresses are stored only as salted hashes. The day rolls over at 00:00 UTC. State
persists to a small JSON file so a process restart keeps it (on hosts whose disk is wiped on
restart, the provider-side spend limit is the backstop).
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

    code = "refused"


class FreeQuotaUsedError(RunRefusedError):
    """This device (or network) has used its free runs: the visitor needs their own key."""

    code = "byok_required"


@dataclass(frozen=True)
class Ticket:
    free: bool  # True: owner's key, charged to the daily budget
    reserved_usd: float


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class SpendGuard:
    def __init__(
        self,
        state_file: Path,
        *,
        daily_usd: float,
        free_runs_per_device: int,
        free_runs_per_ip: int,
        max_concurrent: int,
        reserve_usd: float,
        salt: bytes | None = None,
        clock=_today,
    ) -> None:
        self.state_file = state_file
        self.daily_usd = daily_usd
        self.free_runs_per_device = free_runs_per_device
        self.free_runs_per_ip = free_runs_per_ip
        self.max_concurrent = max_concurrent
        self.reserve_usd = reserve_usd
        self._clock = clock
        self._lock = threading.Lock()
        # A stable salt keeps ids (and their counts) valid across restarts.
        self._salt = salt if salt is not None else secrets.token_bytes(16)
        self._active = 0
        self._reserved = 0.0
        self._day, self._spent, self._devices, self._ips = self._load()

    # -- identity -----------------------------------------------------------------------------

    def ident(self, value: str) -> str:
        return hashlib.sha256(self._salt + value.encode()).hexdigest()[:16]

    # -- persistence --------------------------------------------------------------------------

    def _load(self) -> tuple[str, float, dict[str, int], dict[str, int]]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            devices, ips = dict(data.get("devices", {})), dict(data.get("ips", {}))
            if data.get("day") == self._clock():
                return data["day"], float(data["spent_usd"]), devices, ips
            return self._clock(), 0.0, devices, ips  # new day: budget resets, quotas don't
        except (OSError, ValueError, KeyError, TypeError):
            return self._clock(), 0.0, {}, {}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        payload = {
            "day": self._day,
            "spent_usd": round(self._spent, 6),
            "devices": self._devices,
            "ips": self._ips,
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.state_file)

    def _roll(self) -> None:
        if self._day != self._clock():
            self._day, self._spent = self._clock(), 0.0

    def _free_left(self, device: str | None, ip: str) -> int:
        left = self.free_runs_per_ip - self._ips.get(ip, 0)
        if device is not None:
            left = min(left, self.free_runs_per_device - self._devices.get(device, 0))
        return max(0, left)

    # -- API ----------------------------------------------------------------------------------

    def status(self, device: str | None, ip: str) -> dict:
        with self._lock:
            self._roll()
            remaining = max(0.0, self.daily_usd - self._spent - self._reserved)
            return {
                "free_runs_left": self._free_left(device, ip),
                "free_runs_total": self.free_runs_per_device,
                "budget_open": remaining >= self.reserve_usd,
                "busy": self._active >= self.max_concurrent,
            }

    def admit_free(self, device: str | None, ip: str) -> Ticket:
        with self._lock:
            self._roll()
            if self._free_left(device, ip) <= 0:
                raise FreeQuotaUsedError(
                    f"You've used your {self.free_runs_per_device} free runs. "
                    "Add your own API key to keep going."
                )
            if self._spent + self._reserved + self.reserve_usd > self.daily_usd:
                raise FreeQuotaUsedError(
                    "Today's free budget is used up (it resets at 00:00 UTC). "
                    "Add your own API key to keep going now."
                )
            self._check_capacity()
            if device is not None:
                self._devices[device] = self._devices.get(device, 0) + 1
            self._ips[ip] = self._ips.get(ip, 0) + 1
            self._reserved += self.reserve_usd
            self._active += 1
            self._save()
            return Ticket(free=True, reserved_usd=self.reserve_usd)

    def admit_byok(self) -> Ticket:
        with self._lock:
            self._check_capacity()
            self._active += 1
            return Ticket(free=False, reserved_usd=0.0)

    def _check_capacity(self) -> None:
        if self._active >= self.max_concurrent:
            raise RunRefusedError("The agent is busy with other visitors. Try again shortly.")

    def settle(self, ticket: Ticket, cost_usd: float | None) -> None:
        """Release the run. A free run's real cost is charged to the day; an unknown cost is
        charged as the full reservation, because unknown spend must never look free."""
        with self._lock:
            self._active = max(0, self._active - 1)
            if not ticket.free:
                return
            self._reserved = max(0.0, self._reserved - ticket.reserved_usd)
            self._roll()
            self._spent += ticket.reserved_usd if cost_usd is None else cost_usd
            self._save()
