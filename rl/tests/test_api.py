import json
import pytest
from api.backtest import handle


def test_bad_profile_returns_400():
    status, body = handle({"profile": "nope", "start": "2021-01-01",
                           "end": "2021-06-01", "rebalance": "M"})
    assert status == 400
    assert "error" in body


def test_bad_dates_return_400():
    status, body = handle({"profile": "balanced", "start": "2021-06-01",
                           "end": "2021-01-01", "rebalance": "M"})
    assert status == 400


def test_valid_request_returns_series_and_baselines():
    status, body = handle({"profile": "balanced", "start": "2021-01-01",
                           "end": "2021-12-31", "rebalance": "M"})
    assert status == 200
    assert len(body["agent"]["equity"]) > 100
    assert set(body["baselines"]) == {"equal_weight", "spy"}
    for row in body["agent"]["weights_timeline"]:
        assert abs(sum(row["weights"]) - 1.0) < 1e-6


def test_determinism():
    req = {"profile": "aggressive", "start": "2021-01-01", "end": "2021-12-31", "rebalance": "W"}
    a = handle(req); b = handle(req)
    assert json.dumps(a[1]) == json.dumps(b[1])


def test_agent_and_baselines_share_date_axis():
    status, body = handle({"profile": "balanced", "start": "2021-01-01",
                           "end": "2021-12-31", "rebalance": "M"})
    assert status == 200
    ad = body["agent"]["dates"]
    assert ad == body["baselines"]["equal_weight"]["dates"] == body["baselines"]["spy"]["dates"]
    assert len(ad) == len(body["agent"]["equity"])


@pytest.mark.parametrize("body", [[1, 2, 3], "hello", 42, None])
def test_non_dict_body_returns_400(body):
    from api.backtest import handle
    status, out = handle(body)
    assert status == 400
    assert "error" in out
