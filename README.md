# admission-controller-simulator

An offline simulator of an admission controller for DNN inference on a spatially partitioned GPU (the
Jetson AGX Orin's 16 SMs, MPS static partitions). Requests for profiled networks arrive. The controller
admits or rejects each one and chooses its SMs, checking that every tenant's p99 stays within its
budget. Its decisions are then set against what was actually measured on the board.

```
python3 main.py static     every measured configuration as one admission question, per policy
python3 main.py dynamic    request streams: admissions, wrong ones, over-conservative rejections,
                           SM use and energy, per policy, choice rule, gamma and load
python3 test_sim.py        the selftest
```

Python 3.10+ and the standard library only. Tables go to the terminal, CSVs to `results/`.

## Files

| File | What it does |
|---|---|
| `catalog.py` | reads `data/` into plain dicts: profiles per network and size, model constants, measurements |
| `models.py` | each tenant's predicted p99 slowdown in a configuration (sm-only, constant, l2, ensemble, measured) |
| `energy.py` | the additive power model and the time a set of jobs takes |
| `controller.py` | modes, policies, and the decision: candidate sizes, feasibility, choice rule |
| `simulate.py` | the request stream and the event loop |
| `evaluate.py` | the decisions against the measurements: wrong admissions, over-conservative rejections, unknown |
| `main.py` | the two commands |

## Data

`data/catalog-*.csv` is exported by the runner repository (the measurement and model code, on the
board) with `analysis/catalog.py`. `data/SOURCE` names the runner commit and each file's sha256. The
simulator fits nothing: every number it uses is a profile, a frozen model output or a measurement
from there.

## Modes and policies

A platform runs one mode, chosen for the criticality of its workloads; the modes are never mixed.

- **conservative**: predefined partitions (16, or 8 + 8; the layout list is data, so 8+4+4 joins once
  its models exist). The model is the max of Smooth, Temporal and L2, each with its frozen
  correction, plus a 3-point margin.
- **flexible**: any size in 2-SM steps, up to 16 SMs in total. The model is the normalized L2
  regression, with co-runner traffic × 1.27 and dispatch × g when there are two or more co-runners,
  plus a 3-point margin.
- **Baselines**, on the flexible mode: **sm-only** (interference ignored, as in Zahaf et al.,
  RTNS 2026) and **constant** (the mean measured slowdown, whoever the neighbours are).

Running tenants are never moved. Among the feasible sizes a policy takes the one with the least
predicted energy (`energy`) or the smallest one (`fit`).

## Extending

- a model: one function in `models.py` and one line in `MODELS`;
- a layout or a mode: `LAYOUTS` or `MODES` in `controller.py`;
- a policy: one line in `POLICIES`.

## Not in version 1

- Periodic requests (frame rates). They wait for the periodic measurements, which tell whether the
  back-to-back models bound the GPU time of periodic tenants.
- An MPS-default (unpartitioned) policy.
- Reconfiguration of running tenants.
- Placement of partitions on GPCs.
