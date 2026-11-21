"""The live wiring (memswap/live.py) with the mock mechanisms and judge of live/example_mock.py. No model calls."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live"))
import example_mock  # noqa: E402


def test_live_wiring_routes_profiles_defaults_and_replans():
    r = example_mock.run(verbose=False)
    # the Routing Table gives each profiled type its cheapest mechanism that is as accurate as the best one
    assert r["table"] == {"temporal": "single_lookup", "assistant": "planned_read", "fact": "planned_read"}
    assert r["default"] == "iterative_search"
    # served questions use the mechanism of their type; the unprofiled type goes to m0
    assert [s["mechanism"] for s in r["served"]] == ["single_lookup", "planned_read", "planned_read", "iterative_search"]
    assert [s["known"] for s in r["served"]] == [True, True, True, False]
    assert all(s["tokens"] > 0 for s in r["served"])
    # a lower budget makes the Plan Optimizer move temporal questions to the cheaper mechanism
    assert r["moves"] == [{"type": "temporal", "from": "single_lookup", "to": "planned_read"}]
    assert r["table_low_budget"]["temporal"] == "planned_read"
