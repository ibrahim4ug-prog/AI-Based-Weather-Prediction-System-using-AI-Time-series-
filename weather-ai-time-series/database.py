"""Optional MongoDB storage for weather records (database ``weather_prediction``,
collection ``weather``).

The connection string is read from the ``MONGODB_URI`` environment variable and is
never hard-coded or logged. If MongoDB is not configured, pymongo is missing or the
server is unreachable, :meth:`WeatherDatabase.connect` returns ``False`` and the rest
of the application keeps working in CSV mode.

Safety: user input is never inserted into queries as raw dictionaries. Fields,
sort keys and operators are whitelisted and every value is type-checked, which
prevents operator injection such as ``{"$ne": None}``.
"""
from __future__ import annotations

import math
import numbers
import os
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd

from data_processing import NUMERIC_COLUMNS, TEMPERATURE_COLUMN, VALID_RANGES

try:  # pymongo is optional at import time so the app can run in CSV mode
    from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
    from pymongo.errors import PyMongoError

    PYMONGO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when pymongo is absent
    PYMONGO_AVAILABLE = False
    ASCENDING, DESCENDING = 1, -1
    MongoClient = None  # type: ignore[assignment]
    UpdateOne = None  # type: ignore[assignment]

    class PyMongoError(Exception):  # type: ignore[no-redef]
        """Placeholder so ``except PyMongoError`` is valid without pymongo."""


DB_NAME = "weather_prediction"
COLLECTION_NAME = "weather"
ENV_URI = "MONGODB_URI"
SORTABLE_FIELDS = ["date"] + NUMERIC_COLUMNS
HUMIDITY_LEVELS = ("Low", "Medium", "High")

DateLike = Union[str, date, datetime, pd.Timestamp]


class DatabaseUnavailableError(RuntimeError):
    """MongoDB is not configured, not reachable, or an operation failed."""


# --------------------------------------------------------------------------- #
# Pure validation helpers (no database needed)
# --------------------------------------------------------------------------- #
def to_datetime(value: DateLike) -> datetime:
    """Convert a date-like value to a timezone-naive ``datetime``."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Invalid date value.")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(datetime.combine(value, datetime.min.time()))
    elif isinstance(value, str):
        ts = pd.to_datetime(value.strip(), errors="coerce", utc=True)
        if pd.isna(ts):
            raise ValueError(f"Invalid date: {value!r}.")
    else:
        raise ValueError(f"Unsupported date type: {type(value).__name__}.")
    if pd.isna(ts):
        raise ValueError("Invalid date value.")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise ValueError (rejects bool/str/dict)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"'{name}' must be a finite number.")
    return float(value)


def _check_range(name: str, value: float) -> float:
    low, high = VALID_RANGES[name]
    if not low <= value <= high:
        raise ValueError(f"'{name}' must be between {low:g} and {high:g} (got {value:g}).")
    return value


def validate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a full weather record; returns a clean document.

    ``date`` and ``temperature`` are required; humidity, rainfall, wind_speed and
    pressure are optional. Unknown fields are rejected.
    """
    if not isinstance(record, Mapping):
        raise ValueError("A record must be a mapping of field names to values.")
    unknown = set(record) - set(["date"] + NUMERIC_COLUMNS)
    if unknown:
        raise ValueError("Unknown field(s): " + ", ".join(sorted(map(str, unknown))) + ".")
    if "date" not in record:
        raise ValueError("A record must include 'date'.")
    doc: Dict[str, Any] = {"date": to_datetime(record["date"])}
    for col in NUMERIC_COLUMNS:
        value = record.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if col == TEMPERATURE_COLUMN:
                raise ValueError("A record must include 'temperature'.")
            continue
        doc[col] = _check_range(col, _number(value, col))
    return doc


def validate_updates(updates: Mapping[str, Any]) -> Dict[str, float]:
    """Validate the fields of an update (``date`` cannot be changed)."""
    if not isinstance(updates, Mapping) or not updates:
        raise ValueError("Provide at least one field to update.")
    clean: Dict[str, float] = {}
    for key, value in updates.items():
        if key not in NUMERIC_COLUMNS:
            raise ValueError(f"Field '{key}' cannot be updated. Allowed: {', '.join(NUMERIC_COLUMNS)}.")
        clean[key] = _check_range(key, _number(value, key))
    return clean


def build_query(
    min_temperature: Optional[float] = None,
    max_temperature: Optional[float] = None,
    min_humidity: Optional[float] = None,
    max_humidity: Optional[float] = None,
    min_rainfall: Optional[float] = None,
    max_rainfall: Optional[float] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> Dict[str, Any]:
    """Build a MongoDB filter from whitelisted, validated parameters."""
    query: Dict[str, Any] = {}
    ranges = {
        "temperature": (min_temperature, max_temperature),
        "humidity": (min_humidity, max_humidity),
        "rainfall": (min_rainfall, max_rainfall),
    }
    for field_name, (low, high) in ranges.items():
        clause: Dict[str, float] = {}
        if low is not None:
            clause["$gte"] = _number(low, f"min_{field_name}")
        if high is not None:
            clause["$lte"] = _number(high, f"max_{field_name}")
        if clause:
            if "$gte" in clause and "$lte" in clause and clause["$gte"] > clause["$lte"]:
                raise ValueError(f"min_{field_name} must not exceed max_{field_name}.")
            query[field_name] = clause
    date_clause: Dict[str, datetime] = {}
    if start_date is not None:
        date_clause["$gte"] = to_datetime(start_date)
    if end_date is not None:
        date_clause["$lte"] = to_datetime(end_date)
    if date_clause:
        if "$gte" in date_clause and "$lte" in date_clause and date_clause["$gte"] > date_clause["$lte"]:
            raise ValueError("start_date must not be after end_date.")
        query["date"] = date_clause
    return query


# --------------------------------------------------------------------------- #
# Database wrapper
# --------------------------------------------------------------------------- #
class WeatherDatabase:
    """Thin, safe wrapper around the ``weather`` collection.

    Args:
        uri: connection string; defaults to the ``MONGODB_URI`` environment variable.
        collection: inject an existing collection (used for tests, e.g. mongomock).
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = DB_NAME,
        collection_name: str = COLLECTION_NAME,
        timeout_ms: int = 3000,
        collection: Any = None,
    ) -> None:
        self._uri = (uri if uri is not None else os.environ.get(ENV_URI, "")).strip()
        self.db_name = db_name
        self.collection_name = collection_name
        self.timeout_ms = timeout_ms
        self._client: Any = None
        self._collection: Any = collection
        self.status_message = "Not connected."
        if collection is not None:
            self.status_message = "Connected (injected collection)."
            self._ensure_index()

    # -- connection -------------------------------------------------------- #
    @property
    def is_connected(self) -> bool:
        return self._collection is not None

    def connect(self) -> bool:
        """Try to connect; never raises. Returns True when MongoDB is usable."""
        if self._collection is not None:
            return True
        if not self._uri:
            self.status_message = f"{ENV_URI} is not set, so the app is running in CSV mode."
            return False
        if not PYMONGO_AVAILABLE:
            self.status_message = "pymongo is not installed, so the app is running in CSV mode."
            return False
        try:
            client = MongoClient(self._uri, serverSelectionTimeoutMS=self.timeout_ms)
            client.admin.command("ping")
        except Exception as exc:  # invalid URI, auth, DNS, timeout, ...
            # Only the exception type is reported: messages may contain credentials/hosts.
            self._client = None
            self._collection = None
            self.status_message = (
                f"Could not connect to MongoDB ({type(exc).__name__}); running in CSV mode."
            )
            return False
        self._client = client
        self._collection = client[self.db_name][self.collection_name]
        self.status_message = f"Connected to {self.db_name}.{self.collection_name}."
        self._ensure_index()
        return True

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._collection = None

    def _ensure_index(self) -> None:
        try:
            self._collection.create_index("date", unique=True)
        except Exception:
            self.status_message += " (Could not create the unique date index.)"

    def _col(self) -> Any:
        if self._collection is None:
            raise DatabaseUnavailableError(f"MongoDB is not available. {self.status_message}")
        return self._collection

    def _run(self, func: Callable[[], Any]) -> Any:
        try:
            return func()
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"MongoDB operation failed ({type(exc).__name__}).") from exc

    # -- create ------------------------------------------------------------ #
    def insert_record(self, record: Mapping[str, Any]) -> None:
        """Insert one record. Raises ValueError for invalid data or an existing date."""
        doc = validate_record(record)
        col = self._col()
        if self._run(lambda: col.count_documents({"date": doc["date"]}, limit=1)):
            raise ValueError("A record for this date already exists; use update_record instead.")
        self._run(lambda: col.insert_one(doc))

    def insert_many_records(
        self, records: Union[pd.DataFrame, Iterable[Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        """Upsert many records by date. Invalid rows are skipped and counted.

        Returns ``{"inserted", "updated", "skipped", "errors"}``.
        """
        col = self._col()
        if isinstance(records, pd.DataFrame):
            rows: List[Mapping[str, Any]] = [
                {k: v for k, v in row.items() if not pd.isna(v)}
                for row in records.to_dict("records")
            ]
        else:
            rows = list(records)
        operations = []
        errors: List[str] = []
        skipped = 0
        for i, row in enumerate(rows):
            try:
                doc = validate_record(row)
            except ValueError as exc:
                skipped += 1
                if len(errors) < 5:
                    errors.append(f"row {i}: {exc}")
                continue
            operations.append(UpdateOne({"date": doc["date"]}, {"$set": doc}, upsert=True))
        inserted = updated = 0
        if operations:
            result = self._run(lambda: col.bulk_write(operations, ordered=False))
            inserted, updated = int(result.upserted_count), int(result.modified_count)
        return {"inserted": inserted, "updated": updated, "skipped": skipped, "errors": errors}

    # -- read -------------------------------------------------------------- #
    def get_all_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all records sorted by date (ascending)."""
        return self.find_records(sort_by="date", ascending=True, limit=limit)

    def find_records(
        self,
        min_temperature: Optional[float] = None,
        max_temperature: Optional[float] = None,
        min_humidity: Optional[float] = None,
        max_humidity: Optional[float] = None,
        min_rainfall: Optional[float] = None,
        max_rainfall: Optional[float] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        sort_by: str = "date",
        ascending: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Filter by temperature/humidity/rainfall/date and sort by any numeric field or date."""
        query = build_query(min_temperature, max_temperature, min_humidity, max_humidity,
                            min_rainfall, max_rainfall, start_date, end_date)
        if sort_by not in SORTABLE_FIELDS:
            raise ValueError(f"sort_by must be one of: {', '.join(SORTABLE_FIELDS)}.")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer.")
        col = self._col()

        def _query() -> List[Dict[str, Any]]:
            cursor = col.find(query, {"_id": 0}).sort(sort_by, ASCENDING if ascending else DESCENDING)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)

        return self._run(_query)

    def count_records(self) -> int:
        col = self._col()
        return int(self._run(lambda: col.count_documents({})))

    def fetch_dataframe(self) -> pd.DataFrame:
        """All records as a DataFrame (empty frame with the expected columns if none)."""
        rows = self.get_all_records()
        if not rows:
            return pd.DataFrame(columns=["date"] + NUMERIC_COLUMNS)
        return pd.DataFrame(rows)

    # -- aggregations ------------------------------------------------------ #
    def get_average_temperature(
        self, start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None
    ) -> Optional[float]:
        """Average temperature (°C), or ``None`` if there are no matching records."""
        match = build_query(start_date=start_date, end_date=end_date)
        match["temperature"] = {"$ne": None}
        col = self._col()
        rows = self._run(lambda: list(col.aggregate(
            [{"$match": match}, {"$group": {"_id": None, "avg": {"$avg": "$temperature"}}}])))
        return float(rows[0]["avg"]) if rows and rows[0].get("avg") is not None else None

    def get_total_rainfall(
        self, start_date: Optional[DateLike] = None, end_date: Optional[DateLike] = None
    ) -> float:
        """Total rainfall (mm) over the matching records (0.0 if none)."""
        match = build_query(start_date=start_date, end_date=end_date)
        col = self._col()
        rows = self._run(lambda: list(col.aggregate(
            [{"$match": match}, {"$group": {"_id": None, "total": {"$sum": "$rainfall"}}}])))
        return float(rows[0]["total"]) if rows else 0.0

    def group_by_humidity_level(self, low: float = 40.0, high: float = 70.0) -> List[Dict[str, Any]]:
        """Group by humidity level: Low (< low), Medium (low to < high), High (>= high)."""
        low, high = _number(low, "low"), _number(high, "high")
        if low >= high:
            raise ValueError("low must be smaller than high.")
        level = {"$cond": [{"$lt": ["$humidity", low]}, "Low",
                           {"$cond": [{"$lt": ["$humidity", high]}, "Medium", "High"]}]}
        pipeline = [
            {"$match": {"humidity": {"$ne": None}}},
            {"$group": {"_id": level, "count": {"$sum": 1},
                        "avg_temperature": {"$avg": "$temperature"}}},
        ]
        col = self._col()
        rows = self._run(lambda: list(col.aggregate(pipeline)))
        by_level = {r["_id"]: r for r in rows}
        return [
            {"level": lvl, "count": int(by_level[lvl]["count"]),
             "avg_temperature": by_level[lvl]["avg_temperature"]}
            for lvl in HUMIDITY_LEVELS if lvl in by_level
        ]

    # -- update / delete --------------------------------------------------- #
    def update_record(self, record_date: DateLike, updates: Mapping[str, Any]) -> bool:
        """Update fields of the record on ``record_date``; True if a record matched."""
        clean = validate_updates(updates)
        key = to_datetime(record_date)
        col = self._col()
        result = self._run(lambda: col.update_one({"date": key}, {"$set": clean}))
        return result.matched_count > 0

    def delete_record(self, record_date: DateLike) -> bool:
        """Delete the record on ``record_date``; True if one was deleted."""
        key = to_datetime(record_date)
        col = self._col()
        return self._run(lambda: col.delete_one({"date": key})).deleted_count > 0

    def delete_all_records(self, confirm: bool = False) -> int:
        """Delete every record. Requires ``confirm=True``; returns the number deleted."""
        if confirm is not True:
            raise ValueError("Pass confirm=True to delete all records.")
        col = self._col()
        return int(self._run(lambda: col.delete_many({})).deleted_count)
