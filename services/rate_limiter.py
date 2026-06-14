from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from db.database import Database


@dataclass
class QuotaState:
    queries_count: int
    quota: int
    quota_reset: datetime


class RateLimiter:
    def __init__(self, database: Database, quota: int, window_hours: int):
        self.database = database
        self.quota = quota
        self.window = timedelta(hours=window_hours)

    def check(self, ip: str) -> tuple[bool, QuotaState | None]:
        """Return (allowed, state). state is the doc that would block the
        request when allowed=False, else None."""
        doc = self.database.get_ip_quota(ip)
        if doc is None:
            return True, None
        if _now() >= doc["quota_reset"]:
            return True, None
        if doc["queries_count"] >= doc["quota"]:
            return False, QuotaState(
                queries_count=doc["queries_count"],
                quota=doc["quota"],
                quota_reset=doc["quota_reset"],
            )
        return True, None

    def state(self, ip: str) -> QuotaState:
        doc = self.database.get_ip_quota(ip)
        now = _now()
        if doc is None or now >= doc["quota_reset"]:
            return QuotaState(
                queries_count=0,
                quota=self.quota,
                quota_reset=now + self.window,
            )
        return QuotaState(
            queries_count=doc["queries_count"],
            quota=doc["quota"],
            quota_reset=doc["quota_reset"],
        )

    def record(self, ip: str) -> None:
        doc = self.database.get_ip_quota(ip)
        now = _now()
        if doc is None or now >= doc["quota_reset"]:
            new_count = 1
            new_reset = now + self.window
        else:
            new_count = doc["queries_count"] + 1
            new_reset = doc["quota_reset"]
        self.database.upsert_ip_quota(ip, new_count, self.quota, new_reset)


def _now() -> datetime:
    return datetime.now(timezone.utc)
