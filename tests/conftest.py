import pytest


@pytest.fixture(autouse=True)
def no_browser_during_tests(monkeypatch):
    # Training fixtures should exercise learning, not open browser tabs or servers.
    monkeypatch.setenv("AGIMAC_DASHBOARD", "0")
