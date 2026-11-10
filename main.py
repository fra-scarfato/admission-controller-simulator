"""main.py - the admission-controller simulator.

    python3 main.py static     every measured configuration as one admission question, per policy
    python3 main.py dynamic    request streams: admissions, wrong ones, over-conservative rejections,
                               SM use and energy, per policy, choice rule, gamma and load
    python3 test_sim.py        the selftest

Tables go to the terminal, CSVs to results/. The inputs are data/catalog-*.csv (provenance in
data/SOURCE).

Energy is reported against the same inferences done one network after the other alone on the whole
GPU (the baseline of the tenants experiments): below 1, sharing saved energy. Joules per inference
would mix networks whose inferences differ 30-fold, and every policy admits a different mix.
"""
import csv
import os
import sys
from collections import Counter
from dataclasses import replace

import evaluate
from catalog import load
from controller import GPU_SMS, POLICIES
from simulate import run, workload

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMAS = (1.5, 2, 3, 5)        # D = gamma x p99 alone on 16 SMs
LOADS = (1, 2, 4, 6)           # offered tenants (arrival rate x mean job length)
REQUESTS, SEED = 2000, 1       # every policy sees the same requests for a given gamma and load
MARGINS = range(9)             # points, for the margin curve
WANTED_FROM = ("conservative", "flexible")   # the baselines' admissions are not worth measuring
VERDICTS = ("admit_ok", "wrong_admit", "admit_unknown", "reject_ok", "over_conservative",
            "reject_unknown", "reject_full")


def write(name, rows):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_static(cat):
    rows = evaluate.static(cat, POLICIES, GAMMAS)
    write("static.csv", rows)
    print("Every measured configuration a policy's mode allows; D = gamma x p99 alone on 16 SMs.")
    print("%-13s %-7s %5s %4s %10s %7s %8s %10s" % ("policy", "configs", "gamma", "n", "admit ok", "wrong",
                                                   "over-c.", "reject ok"))
    for name in POLICIES:
        for group in ("built", "unseen"):
            for g in GAMMAS:
                c = evaluate.tally(r for r in rows if (r["policy"], r["group"], r["gamma"]) == (name, group, g))
                if c["n"]:
                    print("%-13s %-7s %5g %4d %10d %7d %8d %10d" % (name, group, g, c["n"], c["admit_ok"],
                          c["wrong_admit"], c["over_conservative"], c["reject_ok"]))
    print("\nMargin curve, all gammas pooled: wrong admissions / over-conservative rejections")
    print("%6s" % "" + "".join("%26s" % n for n in WANTED_FROM))
    print("%6s" % "points" + "%14s | %9s" % ("built", "unseen") * len(WANTED_FROM))
    for m in MARGINS:
        line = "%6d" % m
        for name in WANTED_FROM:
            rs = evaluate.static(cat, {name: replace(POLICIES[name], margin=m / 100)}, GAMMAS)
            b, u = (evaluate.tally(r for r in rs if r["group"] == grp) for grp in ("built", "unseen"))
            line += "%14s | %9s" % ("%d / %d" % (b["wrong_admit"], b["over_conservative"]),
                                     "%d / %d" % (u["wrong_admit"], u["over_conservative"]))
        print(line)


def cmd_dynamic(cat):
    rows, wanted = [], Counter()
    print("%-13s %-6s %5s %4s %8s %6s %7s %8s %6s %6s %7s" % (
        "policy", "choice", "gamma", "load", "admitted", "wrong", "over-c.", "unknown", "full", "SMs", "energy"))
    for g in GAMMAS:
        for ld in LOADS:
            reqs = workload(cat, ld, REQUESTS, g, SEED)
            for name, pol in POLICIES.items():
                for choice in ("energy", "fit"):
                    log, st = run(cat, replace(pol, choice=choice), reqs)
                    c, w = evaluate.judge(cat, log)
                    if name in WANTED_FROM:
                        wanted += w
                    alone16 = sum(n / cat.net[net][GPU_SMS]["rate"] * cat.net[net][GPU_SMS]["W"]
                                  for net, n in st["inferences"].items())
                    row = dict(policy=name, choice=choice, gamma=g, load=ld, requests=len(reqs),
                               admitted=st["admitted"], **{v: c[v] for v in VERDICTS},
                               sm_use=round(st["sm_seconds"] / (GPU_SMS * st["seconds"]), 4),
                               energy_vs_alone16=round(st["joules"] / alone16, 4))
                    rows.append(row)
                    print("%-13s %-6s %5g %4d %7.1f%% %6d %7d %8d %6d %5.0f%% %7.3f" % (
                        name, choice, g, ld, 100 * row["admitted"] / len(reqs), c["wrong_admit"],
                        c["over_conservative"], c["admit_unknown"] + c["reject_unknown"], c["reject_full"],
                        100 * row["sm_use"], row["energy_vs_alone16"]))
    write("dynamic.csv", rows)
    write("wanted.csv", [dict(config=k, decisions=n) for k, n in wanted.most_common()])
    print("\nUnmeasured configurations the conservative and flexible decisions needed most:")
    for k, n in wanted.most_common(15):
        print("  %6d  %s" % (n, k))


if __name__ == "__main__":
    cmd = {"static": cmd_static, "dynamic": cmd_dynamic}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if not cmd:
        sys.exit(__doc__)
    cmd(load(os.path.join(HERE, "data")))
