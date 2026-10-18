"""Same request, same output (apart from timings, cache state and the request id); the seed is echoed."""
import pytest

pytest.importorskip("fastapi")

from conftest import strip_timings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from nanovate.app.api import create_app  # noqa: E402
from nanovate.app.engine import Engine  # noqa: E402

REQ = {"simulate": {"family": "via", "kind": "filament", "seed": 11, "ler_sigma": 1.5, "grain": 0.3, "shift": [2.5, -4.0], "warp": 0.5}}


def test_verify_twice_is_identical(client):
    a, b = (client.post("/api/v1/verify", json=REQ).json() for _ in range(2))
    assert a["seed"] == b["seed"] == 11
    assert strip_timings(a) == strip_timings(b)


def test_simulate_twice_is_byte_identical(client):
    a, b = (client.post("/api/v1/simulate?include_arrays=true", json=REQ["simulate"]).json() for _ in range(2))
    assert strip_timings(a) == strip_timings(b) and a["images"]["sem"]["png"] == b["images"]["sem"]["png"]


def test_fresh_engine_gives_the_same_answer(client, rule_cache):
    """No hidden state: a new engine (cold calibration cache, rules from disk) returns the same response."""
    a = client.post("/api/v1/verify", json=REQ).json()
    b = TestClient(create_app(Engine(rule_cache=rule_cache), frontend=None)).post("/api/v1/verify", json=REQ).json()
    assert strip_timings(a) == strip_timings(b)


def test_input_id_is_a_function_of_the_request(client):
    a = client.post("/api/v1/simulate", json=REQ["simulate"]).json()["input_id"]
    b = client.post("/api/v1/simulate", json={**REQ["simulate"], "seed": 12}).json()["input_id"]
    assert a != b and a == client.post("/api/v1/simulate", json=REQ["simulate"]).json()["input_id"]
    v = client.post("/api/v1/verify", json={"input_id": a}).json()
    assert v["input_id"] == a and v["seed"] == 11
