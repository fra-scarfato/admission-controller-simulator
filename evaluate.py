"""evaluate.py - the decisions against the measurements.

A decision is judged only where the configurations it depends on were measured (catalog-measured.csv;
alone, the measurement is the p99 alone):
  wrong admission    admitted, and some tenant's measured p99 is over its budget
  over-conservative  rejected, though one of its candidate sizes leads to a measured configuration
                     where every tenant is within its budget
  unknown            it cannot be judged: the configuration it led to (admitted), or one of its
                     candidates (rejected, none measured feasible), was never measured
A rejection with no candidate at all (no free SMs in the mode) is a capacity rejection: no model
was asked.

static() turns every measured configuration into one admission question: may these tenants share the
GPU like this, each with budget gamma x its p99 alone on 16 SMs? It simulates nothing, so it is the
direct test of each policy's feasibility check against the data. Configurations are split into those
that built or tuned the models and those no model ever saw (UNSEEN: exp27, the new networks on 8 + 8;
exp36-38, the final check's splits).

judge() scores a simulation's log in the same way, and counts the unmeasured configurations its
decisions needed: the measurements that would settle the most decisions.
"""
from collections import Counter

from catalog import key, parse
from controller import MODES, feasible

UNSEEN = {"exp27", "exp36", "exp37", "exp38"}


def verdict(cat, config, budgets):
    """True if every tenant's measured p99 is within its budget, False if one is over, None if config
    (or one of its tenants) was never measured."""
    if len(config) == 1:
        (net, n), = config
        return cat.net[net][n]["p99_ms"] <= budgets[0]
    m = cat.measured.get(key(config))
    if not m or any(t not in m["p99"] for t in config):
        return None
    return all(m["p99"][t] <= d for t, d in zip(config, budgets))


def static(cat, policies, gammas):
    """One row per policy, measured configuration its mode allows, and gamma."""
    rows = []
    for k, m in sorted(cat.measured.items()):
        config = parse(k)
        group = "unseen" if m["families"] <= UNSEEN else "built"
        for name, pol in policies.items():
            if not MODES[pol.mode]([n for _, n in config]):
                continue
            for g in gammas:
                budgets = [g * cat.net[net][16]["p99_ms"] for net, _ in config]
                truth = verdict(cat, config, budgets)
                if truth is not None:
                    rows.append(dict(policy=name, group=group, gamma=g, config=k,
                                     admit=feasible(cat, pol, config, budgets), truth=truth))
    return rows


def tally(rows):
    """Counts of static() rows: n, admitted right, wrong admissions, over-conservative, rejected right."""
    c = Counter()
    for r in rows:
        c["n"] += 1
        c[("admit_ok" if r["truth"] else "wrong_admit") if r["admit"] else
          ("over_conservative" if r["truth"] else "reject_ok")] += 1
    return c


def judge(cat, log):
    """(Counter of a simulation log's decisions by verdict, Counter of the unmeasured configurations
    they needed)."""
    c, wanted = Counter(), Counter()
    for d in log:
        config = [(n, s) for n, s, _ in d["before"]]
        budgets = [D for _, _, D in d["before"]] + [d["D"]]
        if d["sms"] is not None:
            after = config + [(d["net"], d["sms"])]
            v = verdict(cat, after, budgets)
            c[{True: "admit_ok", False: "wrong_admit", None: "admit_unknown"}[v]] += 1
            if v is None:
                wanted[key(after)] += 1
        elif not d["cands"]:
            c["reject_full"] += 1
        else:
            vs = {key(config + [(d["net"], n)]): verdict(cat, config + [(d["net"], n)], budgets) for n in d["cands"]}
            if True in vs.values():
                c["over_conservative"] += 1
            elif None in vs.values():
                c["reject_unknown"] += 1
                wanted.update(k for k, v in vs.items() if v is None)
            else:
                c["reject_ok"] += 1
    return c, wanted
