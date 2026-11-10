"""controller.py - admit or reject one request, and choose its SMs.

A platform runs one mode, chosen for the criticality of the workloads it hosts. Modes are never mixed
on one GPU.
  conservative  predefined partitions: the sizes in use must stay part of one of LAYOUTS (for now 16,
                or 8 + 8; 8+4+4 joins once the stress-sensitivity models exist on 4 SMs)
  flexible      any size the catalog has (2-SM steps), as long as the sizes add up to at most 16 SMs

A policy is a mode, a model (models.py), a margin in slowdown units (0.03 = 3 points) and a choice rule
among the feasible sizes:
  energy  the least predicted energy to finish the running jobs and the newcomer's (energy.finish):
          the professor's rule
  fit     the smallest feasible size, which keeps SMs for later arrivals
A size is feasible when every tenant, the newcomer and each running one, is predicted within its p99
budget D. Running tenants are never moved, because MPS static must restart a client to move it. So a
request is judged against the partitions as they are, and when a tenant leaves its SMs are free.

A new mode is one entry in MODES; a new policy is one line in POLICIES.
"""
from collections import Counter
from dataclasses import dataclass

import energy
from models import MODELS

GPU_SMS = 16
LAYOUTS = [(16,), (8, 8)]
MODES = {"conservative": lambda sizes: any(Counter(sizes) <= Counter(lay) for lay in LAYOUTS),
         "flexible": lambda sizes: sum(sizes) <= GPU_SMS}


@dataclass(frozen=True)
class Policy:
    mode: str
    model: str
    margin: float
    choice: str = "energy"


# The two modes the professor proposed, each with the safety the user chose (2026-11-10), and the two
# simple baselines, which run on the flexible mode: SM availability only, and interference counted
# blindly.
POLICIES = {"conservative": Policy("conservative", "ensemble", 0.03),
            "flexible": Policy("flexible", "l2", 0.03),
            "sm-only": Policy("flexible", "sm-only", 0.0),
            "constant": Policy("flexible", "constant", 0.0)}


@dataclass
class Request:
    t: float              # arrival time (s)
    net: str
    n_inf: float          # the job: how many inferences
    D: float              # p99 budget (ms)
    sms: int = None       # the SMs it asks for, or None to let the controller choose


def predicted_p99(cat, policy, config):
    """Each tenant's predicted p99 (ms), None where the model has no answer. Alone, no margin: the p99
    alone is measured."""
    model, out = MODELS[policy.model], []
    for i, (net, n) in enumerate(config):
        s = model(cat, config, i)
        margin = policy.margin if len(config) > 1 else 0.0
        out.append(None if s is None else cat.net[net][n]["p99_ms"] * (s + margin))
    return out


def feasible(cat, policy, config, budgets):
    return all(p is not None and p <= d for p, d in zip(predicted_p99(cat, policy, config), budgets))


def candidates(cat, policy, taken, sms=None):
    """The sizes a newcomer may take beside the sizes already taken, in the policy's mode."""
    return [n for n in ([sms] if sms else cat.sizes) if MODES[policy.mode](list(taken) + [n])]


def decide(cat, policy, running, req):
    """The SMs req gets, or None if it is rejected. running: the tenants on the GPU (net, sms, D, left)."""
    config, budgets = [(t.net, t.sms) for t in running], [t.D for t in running] + [req.D]
    ok = [n for n in candidates(cat, policy, [s for _, s in config], req.sms)
          if feasible(cat, policy, config + [(req.net, n)], budgets)]
    if not ok:
        return None
    if policy.choice == "fit":
        return min(ok)
    jobs = [(t.net, t.sms, t.left) for t in running]
    return min(ok, key=lambda n: energy.finish(cat, jobs + [(req.net, n, req.n_inf)])[1])
