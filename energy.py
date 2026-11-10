"""energy.py - the additive power model and the time a set of jobs takes (the runner's energy model).

While a configuration S runs, each network back to back on its partition, the module draws

    P(S) = sum over S of P_alone(network, its SMs) - (|S| - 1) B

where B is the power drawn once however many networks run (16.7 W, fitted on the 30 s pairs). With k
others running, a network goes at u_k times its rate alone: u_1 from the pairs, u_2 from the 8+4+4
triples, and u_k = u_2 for k > 2, an extrapolation no run has checked. In the runner, on runs it never
used, this model got the phase power within 2-4 % and the totals within +0.3 to +7.3 %, and it picked
the cheapest allocation in 13 of 14 sets.

ponytail: an empty GPU draws nothing here. The module's idle power with the clocks locked was never
measured. It adds the same joules per second to every policy, but not the same per inference.
"""


def power(cat, config):
    if not config:
        return 0.0
    return sum(cat.net[n][s]["W"] for n, s in config) - (len(config) - 1) * cat.const["B"]


def rates(cat, config):
    """Inferences per second of each tenant of config, all running."""
    k = len(config) - 1
    u = 1.0 if k == 0 else cat.const["u1"] if k == 1 else cat.const["u2"]
    return [cat.net[n][s]["rate"] * u for n, s in config]


def finish(cat, jobs):
    """(seconds, joules) until jobs [(net, SMs, inferences left)], side by side from now, are all done.
    A finished job's partition stays idle: nobody moves into it."""
    left = [list(j) for j in jobs]
    t = e = 0.0
    while left:
        config = [(n, s) for n, s, _ in left]
        r = rates(cat, config)
        dt = min(x[2] / ri for x, ri in zip(left, r))
        t, e = t + dt, e + dt * power(cat, config)
        for x, ri in zip(left, r):
            x[2] -= ri * dt
        left = [x for x in left if x[2] > 1e-6]
    return t, e
