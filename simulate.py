"""simulate.py - requests arrive, the controller decides, the admitted jobs run and leave.

    log, stats = run(cat, policy, workload(cat, load, n, gamma, seed))

A request is a job: a network, a number of inferences and a p99 budget D. Admitted jobs run back to
back (the saturated case every model was built on) at the energy model's rates, and leave when done.
Between two events the rates and the power are constant, so time and energy are exact. The log keeps
every decision with what the controller saw: the tenants already running (net, SMs, D), the request,
its candidate sizes and the size it got. evaluate.py sets that against the measurements.

workload() draws the requests:
- Poisson arrivals;
- networks uniform over the catalog;
- each job long enough for an exponential time (mean MEAN_S) alone on the whole GPU, so a RetinaNet
  job and a MobileNet job last alike;
- D = gamma x the network's p99 alone on 16 SMs.
load is the offered number of tenants: arrival rate x MEAN_S.

ponytail: saturated jobs only. Periodic requests (a frame rate) wait for the periodic runs (stage A/B),
which tell whether the back-to-back models bound the GPU time of periodic tenants.
"""
import random
from dataclasses import dataclass

import energy
from controller import GPU_SMS, Request, candidates, decide

MEAN_S = 60.0


@dataclass
class Tenant:
    net: str
    sms: int
    D: float
    left: float           # inferences still to do


def workload(cat, load, n, gamma, seed):
    rng, nets, t, out = random.Random(seed), sorted(cat.net), 0.0, []
    for _ in range(n):
        t += rng.expovariate(load / MEAN_S)
        net = rng.choice(nets)
        whole = cat.net[net][GPU_SMS]
        out.append(Request(t, net, whole["rate"] * rng.expovariate(1 / MEAN_S), gamma * whole["p99_ms"]))
    return out


def run(cat, policy, requests):
    """(log, stats) of one policy on one request stream. stats: seconds until the last job ends, joules,
    inferences done per network, SM-seconds allocated, admitted and rejected counts."""
    running, log = [], []
    st = dict(seconds=0.0, joules=0.0, sm_seconds=0.0, admitted=0, rejected=0, inferences={})

    def advance(until):
        # Run the jobs until `until`, stepping from one departure to the next.
        while running and st["seconds"] < until:
            config = [(x.net, x.sms) for x in running]
            r = energy.rates(cat, config)
            dt = min(min(x.left / ri for x, ri in zip(running, r)), until - st["seconds"])
            st["seconds"] += dt
            st["joules"] += dt * energy.power(cat, config)
            st["sm_seconds"] += dt * sum(x.sms for x in running)
            for x, ri in zip(running, r):
                x.left -= ri * dt
                st["inferences"][x.net] = st["inferences"].get(x.net, 0.0) + ri * dt
            running[:] = [x for x in running if x.left > 1e-6]
        if until != float("inf"):
            st["seconds"] = max(st["seconds"], until)     # an empty GPU waits for the next arrival

    for req in requests:
        advance(req.t)
        before = [(x.net, x.sms, x.D) for x in running]
        sms = decide(cat, policy, running, req)
        log.append(dict(t=req.t, net=req.net, D=req.D, before=before, sms=sms,
                        cands=candidates(cat, policy, [s for _, s, _ in before], req.sms)))
        if sms is None:
            st["rejected"] += 1
        else:
            st["admitted"] += 1
            running.append(Tenant(req.net, sms, req.D, req.n_inf))
    advance(float("inf"))
    return log, st
