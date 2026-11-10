"""models.py - each tenant's predicted p99 slowdown in a configuration.

A model is a function model(cat, config, i). It returns the p99 slowdown of tenant i of config (a list
of (net, SMs)) over that network alone on the same SMs, or None when the model has no answer there.
The controller turns it into a p99: the tenant's p99 alone at its size times (slowdown + margin). The
margin belongs to the policy (a criticality level), never to a model.

  sm-only   1: interference ignored, only the free SMs count (the admission of Zahaf et al.,
            RTNS 2026: a free partition big enough)
  constant  the mean measured slowdown of the 12 development pairs on 8 + 8, whoever the neighbours
            are: interference counted, but blind to who causes it
  l2        the normalized L2 regression (the runner's geometry.normalized). The tenant's sensitivity
            is read from its L2 traffic per SM, scaled to the 8 SMs it was fitted on, and multiplies
            its co-runners' L2 traffic. With two or more co-runners their traffic is summed and
            multiplied by F, and the presence by g (the dispatch of several clients).
  ensemble  the conservative one, defined only where all its members are: alone, or on 8 + 8, where
            it is the max of Smooth, Temporal and L2, each shifted by its frozen correction (the
            runner's ensemble.py)
  measured  the oracle: the measured slowdown, where the configuration was measured

Alone, every model answers 1: the p99 alone is measured, not predicted. A new model is one function
here plus one line in MODELS.
"""
from catalog import key

F = 1.27   # two or more co-runners: their summed L2 traffic times F (thesis tab:stress-triple-norm;
           # chosen by the user for both modes, 2026-11-10)


def sm_only(cat, config, i):
    return 1.0


def constant(cat, config, i):
    return 1.0 if len(config) == 1 else cat.const["constant"]


def l2(cat, config, i, f=F):
    if len(config) == 1:
        return 1.0
    net, n = config[i]
    me, c = cat.net[net][n], cat.const
    load = sum(cat.net[o][m]["L"] for j, (o, m) in enumerate(config) if j != i)
    many = len(config) > 2
    sensitivity = c["b0"] + c["b1"] * 8.0 * me["L"] / n
    return 1 + (c["g"] if many else 1.0) * me["p"] + sensitivity * (f if many else 1.0) * load


def ensemble(cat, config, i):
    if len(config) == 1:
        return 1.0
    if len(config) != 2 or any(n != 8 for _, n in config):
        return None
    m, c = cat.pair.get((config[i][0], config[1 - i][0])), cat.const
    if m is None:
        return None
    return max(m["smooth"] + c["shift_smooth"], m["temporal"] + c["shift_temporal"], m["l2"] + c["shift_l2"])


def measured(cat, config, i):
    if len(config) == 1:
        return 1.0
    m = cat.measured.get(key(config))
    p99 = m and m["p99"].get(config[i])
    return p99 / cat.net[config[i][0]][config[i][1]]["p99_ms"] if p99 else None


MODELS = {"sm-only": sm_only, "constant": constant, "l2": l2, "ensemble": ensemble, "measured": measured}
