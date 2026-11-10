"""test_sim.py - the selftest: a two-network catalog small enough to check by hand.

    python3 test_sim.py
"""
from dataclasses import replace

import energy
import evaluate
from catalog import Catalog, key, parse
from controller import MODES, POLICIES, Request, decide
from models import ensemble, l2, measured
from simulate import run

# a: 1.0 ms on 8 SMs, 0.6 on 16; b: 2.0 and 1.2. Measured together on 8 + 8: 1.4 and 2.3 ms.
CAT = Catalog(
    net={"a": {8: dict(p99_ms=1.0, p=0.1, L=100.0, rate=100.0, W=20.0),
               16: dict(p99_ms=0.6, p=None, L=None, rate=160.0, W=28.0)},
         "b": {8: dict(p99_ms=2.0, p=0.05, L=200.0, rate=50.0, W=25.0),
               16: dict(p99_ms=1.2, p=None, L=None, rate=80.0, W=33.0)}},
    pair={("a", "b"): dict(smooth=1.3, temporal=1.2, l2=1.25), ("b", "a"): dict(smooth=1.1, temporal=1.15, l2=1.12)},
    const=dict(b0=0.002, b1=-1e-5, g=1.5, shift_smooth=0.0, shift_temporal=0.0, shift_l2=0.0,
               constant=1.25, B=15.0, u1=0.8, u2=0.7),
    measured={"a@8+b@8": {"p99": {("a", 8): 1.4, ("b", 8): 2.3}, "families": {"exp24"}}})


def main():
    assert parse(key([("b", 8), ("a", 4)])) == [("a", 4), ("b", 8)]
    con, flex = MODES["conservative"], MODES["flexible"]
    assert con([8]) and con([8, 8]) and con([16]) and not con([16, 8]) and not con([4])
    assert flex([2, 14]) and flex([8, 8]) and not flex([10, 8])

    # l2: sensitivity 0.002 - 1e-5 * 8 * 100 / 8 = 0.001, times b's 200 GB/s, plus presence 0.1;
    # with two co-runners, presence x g and the load x F
    ab = [("a", 8), ("b", 8)]
    assert abs(l2(CAT, ab, 0) - 1.3) < 1e-12
    assert abs(l2(CAT, ab + [("b", 8)], 0, f=1.27) - (1 + 1.5 * 0.1 + 0.001 * 1.27 * 400)) < 1e-12
    assert ensemble(CAT, ab, 0) == 1.3 and ensemble(CAT, [("a", 4), ("b", 12)], 0) is None
    assert ensemble(CAT, [("a", 16)], 0) == 1.0
    assert abs(measured(CAT, ab, 1) - 1.15) < 1e-12 and measured(CAT, [("a", 8), ("a", 8)], 0) is None

    # Alone on an empty GPU: D 0.7 fits only on 16; D 1.5 fits on both, fit takes 8, energy takes 16
    # (0.175 J per inference against 0.2)
    pol = POLICIES["conservative"]
    assert decide(CAT, pol, [], Request(0, "a", 100, 0.7)) == 16
    assert decide(CAT, replace(pol, choice="fit"), [], Request(0, "a", 100, 1.5)) == 8
    assert decide(CAT, pol, [], Request(0, "a", 100, 1.5)) == 16
    # Together at u1 = 0.8: a (80/s) ends at 1.25 s at 20 + 25 - 15 W; b has done 50, its other 50
    # alone take 1 s at 25 W.
    assert abs(energy.finish(CAT, [("a", 8, 100), ("b", 8, 100)])[1] - (30 * 1.25 + 25 * 1.0)) < 1e-9

    # A departure frees the SMs: a (1 s alone on 8) is gone when b arrives at 10 s
    fit = replace(pol, choice="fit")
    log, st = run(CAT, fit, [Request(0, "a", 100, 1.5), Request(10, "b", 50, 3.0)])
    assert [d["sms"] for d in log] == [8, 8] and log[1]["before"] == []
    assert abs(st["seconds"] - 11) < 1e-9 and abs(st["joules"] - 45) < 1e-9 and abs(st["sm_seconds"] - 16) < 1e-9

    # Against the measurement: sm-only admits b beside a with D 2.2, measured 2.3 -> wrong admission; the
    # ensemble predicts b at 2.0 x (1.15 + 0.03) = 2.36 > 2.35 and rejects, measured 2.3 -> over-conservative
    reqs = lambda d: [Request(0, "a", 1000, 1.5), Request(0.5, "b", 1000, d)]
    c, wanted = evaluate.judge(CAT, run(CAT, replace(POLICIES["sm-only"], choice="fit"), reqs(2.2))[0])
    assert c["wrong_admit"] == 1 and c["admit_ok"] == 1 and not wanted
    c, _ = evaluate.judge(CAT, run(CAT, fit, reqs(2.35))[0])
    assert c["over_conservative"] == 1
    print("  selftest passed")


if __name__ == "__main__":
    main()
