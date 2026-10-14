from datetime import datetime

import pytest

import database
from database import (
    DatabaseUnavailableError,
    WeatherDatabase,
    build_query,
    validate_record,
    validate_updates,
)


# ---- fallback behaviour ---------------------------------------------------- #
def test_no_uri_means_csv_mode(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    db = WeatherDatabase()
    assert db.connect() is False
    assert db.is_connected is False
    assert "CSV mode" in db.status_message


def test_operations_raise_clear_error_when_unavailable(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    db = WeatherDatabase()
    db.connect()
    with pytest.raises(DatabaseUnavailableError):
        db.get_all_records()
    with pytest.raises(DatabaseUnavailableError):
        db.insert_record({"date": "2024-01-01", "temperature": 20})


def test_unreachable_server_does_not_crash():
    db = WeatherDatabase(uri="mongodb://127.0.0.1:1/?connectTimeoutMS=200", timeout_ms=300)
    assert db.connect() is False
    assert "CSV mode" in db.status_message


def test_malformed_uri_does_not_crash():
    db = WeatherDatabase(uri="this is not a uri", timeout_ms=200)
    assert db.connect() is False


def test_status_message_never_contains_credentials():
    db = WeatherDatabase(uri="mongodb://user:secretpw@127.0.0.1:1/", timeout_ms=300)
    db.connect()
    assert "secretpw" not in db.status_message


# ---- validation (no server needed) ---------------------------------------- #
def test_validate_record_ok():
    doc = validate_record({"date": "2024-01-02", "temperature": 20, "humidity": 55.5})
    assert doc == {"date": datetime(2024, 1, 2), "temperature": 20.0, "humidity": 55.5}


@pytest.mark.parametrize("bad", [
    {"temperature": 20},                                   # no date
    {"date": "2024-01-01"},                                # no temperature
    {"date": "garbage", "temperature": 20},                # bad date
    {"date": "2024-01-01", "temperature": "hot"},          # non-numeric
    {"date": "2024-01-01", "temperature": 500},            # out of range
    {"date": "2024-01-01", "temperature": 20, "humidity": 120},
    {"date": "2024-01-01", "temperature": 20, "evil": 1},  # unknown field
    {"date": "2024-01-01", "temperature": {"$gt": 0}},     # operator injection
])
def test_validate_record_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_record(bad)


def test_validate_updates():
    assert validate_updates({"temperature": 21}) == {"temperature": 21.0}
    for bad in ({}, {"date": "2024-01-01"}, {"temperature": "x"}, {"humidity": 101}):
        with pytest.raises(ValueError):
            validate_updates(bad)


def test_build_query_blocks_operator_injection():
    with pytest.raises(ValueError):
        build_query(min_temperature={"$ne": None})
    with pytest.raises(ValueError):
        build_query(min_temperature=30, max_temperature=10)
    with pytest.raises(ValueError):
        build_query(start_date="2024-02-01", end_date="2024-01-01")
    assert build_query(min_rainfall=1, start_date="2024-01-01") == {
        "rainfall": {"$gte": 1.0}, "date": {"$gte": datetime(2024, 1, 1)}}


# ---- CRUD against an in-memory MongoDB (mongomock) ------------------------ #
@pytest.fixture
def db():
    pytest.importorskip("pymongo")
    mongomock = pytest.importorskip("mongomock")
    collection = mongomock.MongoClient()["weather_prediction"]["weather"]
    return WeatherDatabase(collection=collection)


@pytest.fixture
def seeded(db):
    rows = [
        {"date": "2024-01-01", "temperature": 10.0, "humidity": 30.0, "rainfall": 0.0},
        {"date": "2024-01-02", "temperature": 20.0, "humidity": 55.0, "rainfall": 5.0},
        {"date": "2024-01-03", "temperature": 30.0, "humidity": 85.0, "rainfall": 10.0},
        {"date": "2024-01-04", "temperature": 40.0, "humidity": 90.0, "rainfall": 2.5},
    ]
    summary = db.insert_many_records(rows)
    assert summary["inserted"] == 4 and summary["skipped"] == 0
    return db


def test_insert_and_get_all(db):
    db.insert_record({"date": "2024-01-01", "temperature": 12.5})
    records = db.get_all_records()
    assert len(records) == 1 and records[0]["temperature"] == 12.5 and "_id" not in records[0]


def test_duplicate_date_rejected(db):
    db.insert_record({"date": "2024-01-01", "temperature": 12.5})
    with pytest.raises(ValueError, match="already exists"):
        db.insert_record({"date": "2024-01-01", "temperature": 13.0})


def test_bulk_upsert_updates_and_skips_invalid(seeded):
    summary = seeded.insert_many_records([
        {"date": "2024-01-01", "temperature": 11.0, "humidity": 30.0, "rainfall": 0.0},
        {"date": "2024-01-09", "temperature": 99999},
    ])
    assert summary["updated"] == 1 and summary["skipped"] == 1 and summary["errors"]
    assert seeded.count_records() == 4


def test_filtering_and_sorting(seeded):
    hot = seeded.find_records(min_temperature=25)
    assert [r["temperature"] for r in hot] == [30.0, 40.0]
    wet = seeded.find_records(min_rainfall=2.5, max_rainfall=5.0, sort_by="temperature", ascending=False)
    assert [r["temperature"] for r in wet] == [40.0, 20.0]
    humid = seeded.find_records(min_humidity=50, max_humidity=88)
    assert len(humid) == 2
    ranged = seeded.find_records(start_date="2024-01-02", end_date="2024-01-03")
    assert len(ranged) == 2
    assert len(seeded.find_records(limit=1)) == 1


def test_invalid_sort_and_limit(seeded):
    with pytest.raises(ValueError):
        seeded.find_records(sort_by="$where")
    with pytest.raises(ValueError):
        seeded.find_records(limit=0)


def test_aggregations(seeded):
    assert seeded.get_average_temperature() == pytest.approx(25.0)
    assert seeded.get_total_rainfall() == pytest.approx(17.5)
    groups = {g["level"]: g for g in seeded.group_by_humidity_level()}
    assert groups["Low"]["count"] == 1
    assert groups["Medium"]["count"] == 1
    assert groups["High"]["count"] == 2
    assert groups["High"]["avg_temperature"] == pytest.approx(35.0)


def test_aggregations_on_empty_collection(db):
    assert db.get_average_temperature() is None
    assert db.get_total_rainfall() == 0.0
    assert db.group_by_humidity_level() == []


def test_update_and_delete(seeded):
    assert seeded.update_record("2024-01-02", {"temperature": 22.0}) is True
    assert seeded.update_record("2030-01-01", {"temperature": 22.0}) is False
    assert seeded.find_records(start_date="2024-01-02", end_date="2024-01-02")[0]["temperature"] == 22.0
    assert seeded.delete_record("2024-01-02") is True
    assert seeded.delete_record("2024-01-02") is False
    assert seeded.count_records() == 3


def test_delete_all_requires_confirmation(seeded):
    with pytest.raises(ValueError):
        seeded.delete_all_records()
    assert seeded.delete_all_records(confirm=True) == 4


def test_fetch_dataframe(seeded):
    df = seeded.fetch_dataframe()
    assert len(df) == 4 and "temperature" in df.columns
