"""catalog.py - what the controller knows: profiles, model constants, measurements.

The files in data/ are exported by the runner repository (analysis/catalog.py), which measures the
networks on the Jetson AGX Orin and fits the models. data/SOURCE names the runner commit and each
file's sha256. This module only reads them into plain dicts, so every number the simulator uses has
one origin.

    cat = load("data")
    cat.net["mobilenet"][8]                MobileNet alone on 8 SMs: p99_ms, p, L, rate, W
    cat.pair[("mobilenet", "resnet")]      the 8 + 8 ensemble members: smooth, temporal, l2
    cat.const["B"]                         a model constant (catalog-constants.csv)
    cat.measured["mobilenet@8+resnet@8"]   {"p99": {(net, SMs): ms}, "families": {...}}

A configuration is a list of tenants (net, SMs); key() names it whatever the order of its tenants.
"""
import csv
import os
from dataclasses import dataclass


def key(config):
    return "+".join(sorted("%s@%d" % t for t in config))


def parse(k):
    """key() backwards: "a@4+b@12" -> [("a", 4), ("b", 12)]."""
    return [(net, int(n)) for net, n in (t.split("@") for t in k.split("+"))]


@dataclass
class Catalog:
    net: dict        # {net: {SMs: {"p99_ms", "p", "L", "rate", "W"}}}; p and L are None on 16 SMs
    pair: dict       # {(target, co-runner): {"smooth", "temporal", "l2"}}, raw p99 slowdowns on 8 + 8
    const: dict      # {name: value}
    measured: dict   # {key: {"p99": {(net, SMs): ms}, "families": set of runner experiment keys}}

    @property
    def sizes(self):
        """The partition sizes every network was profiled at, ascending."""
        return sorted(set.intersection(*(set(v) for v in self.net.values())))


def load(datadir):
    rows = lambda name: list(csv.DictReader(open(os.path.join(datadir, "catalog-%s.csv" % name))))
    num = lambda x: float(x) if x != "" else None
    net = {}
    for r in rows("networks"):
        net.setdefault(r["net"], {})[int(r["sms"])] = {k: num(r[k]) for k in ("p99_ms", "p", "L", "rate", "W")}
    pair = {(r["target"], r["corunner"]): {k: float(r[k]) for k in ("smooth", "temporal", "l2")}
            for r in rows("pairs88")}
    const = {r["name"]: float(r["value"]) for r in rows("constants")}
    measured = {}
    for r in rows("measured"):
        m = measured.setdefault(r["config"], {"p99": {}, "families": set()})
        t = (r["net"], int(r["sms"]))
        # A configuration measured more than once keeps each tenant's worst p99: it is feasible only if
        # every measurement of it met the budget.
        m["p99"][t] = max(m["p99"].get(t, 0.0), float(r["p99_ms"]))
        m["families"].add(r["family"])
    return Catalog(net, pair, const, measured)
