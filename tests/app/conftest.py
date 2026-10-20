"""Shared fixtures for the app tests. Each test module skips itself without the optional [app] extra."""
import pytest

try:
    from fastapi.testclient import TestClient

    from nanovate.app.api import create_app
    from nanovate.app.engine import Engine
except ImportError:   # research install without [app]: the test modules importorskip and never use these
    TestClient = create_app = Engine = None


@pytest.fixture(scope="session")
def rule_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("app_cache") / "d2d_rules.json"


@pytest.fixture(scope="session")
def engine(rule_cache):
    """One v2 engine for the session (the existing parity / UI tests pin v2); its first dose-50 request fits the D2D
    rule (~20 s) and writes ``rule_cache``."""
    return Engine(rule_cache=rule_cache, version="v2")


@pytest.fixture(scope="session")
def client(engine):
    return TestClient(create_app(engine, frontend=None), raise_server_exceptions=False)


@pytest.fixture(scope="session")
def rule_cache_v3(tmp_path_factory):
    return tmp_path_factory.mktemp("app_cache_v3") / "d2d_rules.json"


@pytest.fixture(scope="session")
def engine_v3(rule_cache_v3):
    """The default (v3) engine; its own rule cache so the two engines never evict each other's entries."""
    return Engine(rule_cache=rule_cache_v3, version="v3")


@pytest.fixture(scope="session")
def client_v3(engine_v3):
    return TestClient(create_app(engine_v3, frontend=None), raise_server_exceptions=False)


STRIP = {"request_id", "ms", "timings_ms", "runtime"}


def strip_timings(x):
    """Response without the fields allowed to differ between identical requests (timings, cache state, request id)."""
    if isinstance(x, dict):
        return {k: strip_timings(v) for k, v in x.items() if k not in STRIP}
    if isinstance(x, list):
        return [strip_timings(v) for v in x]
    return x
