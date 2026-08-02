# Reward-weight tuning

How the five penalty weights in `cat_env/cat_env.py` were set, and the sweep that
set them. Run with `docs/sweep.py`; every number below is 500 random drops of the
privileged teacher via `docs/evaluate.py`. **74 training runs (+3 outside the harness), 77M environment
steps**, both variants, archived in `docs/reward_tuning_runs.jsonl`.

## Why this study was redone from scratch

The previous version of this document was written against a different simulator.
Two things changed underneath it:

- **The sim2real fix** (deadband, minimum-PWM floor, encoder quantization in the
  inner loop) made the task materially harder. The teacher shipped with that
  commit scores **33.3%** under the current physics against the **62.8%** recorded
  for it in `docs/EVALUATION.md`. It landed in the *same commit* that added this
  document, so the tables described an env that no longer existed.
- **The teacher observation grew** from 25 to 73 dims with the privileged
  domain-randomization block (`cat_env.py::DR_DIM`).

The grid is sized in reward units anchored to measured baseline magnitudes, so
every weight in the old study was derived from numbers that no longer hold. The
old results were deleted rather than amended.

## Method

**Start from zero.** All five penalties (`w_sm`, `w_en`, `w_av`, `w_jv`, `w_time`)
set to 0, leaving only `w_pos` and `w_bonus`. That baseline is the reference: the
best success the task reward alone reaches, and the magnitude of each penalized
quantity when nothing discourages it.

**Raise one weight at a time until success drops.** Each term swept alone, other
four at zero, over a 4-point geometric grid, then combined.

**Size the grid in reward units, not raw weights.** A raw weight is meaningless on
its own: `m_av` ~ 18 and `m_sm` ~ 0.25, so `w = 0.1` is crushing for one and
negligible for the other. Each grid point is chosen so that `w * m_baseline` equals
a fixed **fraction of that variant's baseline task reward**: 2%, 6%, 18%, 50%. A
given grid position then means the same thing for every term *and* for both
variants, which is what makes the two knees comparable.

**Read the change in the term, not its absolute value.** The env reports each
penalty's *unweighted* magnitude as `m_*` in `info` beside the weighted `r_*`
(`docs/evaluate.py --stats`). `r_*` is zero at `w = 0` and confounds weight with
behavior everywhere else, so it cannot answer "did this weight change what the
robot does?". `m_*` can, and with far less variance than success rate.

**Noise floor, measured directly.** Every baseline and every shipping candidate ran
3 seeds. Seed sd is **1.8–4.4 pp** (tail) and **0.8–3.5 pp** (no-tail), against the
±2.2 pp sampling error of a 500-drop evaluation. A single-run success difference is
treated as real only past ~5 pp **and** with the magnitude moved **and** with median
tilt moving too. Multi-seed differences are judged on the standard error of the
difference.

## Does the privileged DR block earn its place?

The teacher sees this episode's actual randomization draw; the student never does.
6 control runs trained on the pre-DR 25-dim observation, everything else identical,
all penalties at zero, 3 seeds per cell:

| variant | obs | success | `m_sm` | `m_en` | `m_av` | `m_jv` | `m_time` |
|---|---|---:|---:|---:|---:|---:|---:|
| tail | 25-dim control | 49.7% ±4.4 | 0.340 | 0.480 | 18.2 | 141 | 0.482 |
| tail | 73-dim privileged | 47.3% ±1.8 | **0.250** | 0.464 | 18.3 | **123** | **0.353** |
| | | −2.3 pp | −27% | −3% | +0% | −13% | −27% |
| no-tail | 25-dim control | 28.7% ±2.7 | 0.251 | 0.196 | 20.8 | 152 | 0.384 |
| no-tail | 73-dim privileged | 30.4% ±3.5 | **0.200** | 0.184 | 20.4 | **130** | **0.304** |
| | | +1.7 pp | −20% | −6% | −2% | −14% | −21% |

**Success is a wash** (−2.3 pp one way, +1.7 pp the other, both inside seed spread).
**Motion is not.** Tail `m_sm` reads 0.245/0.249/0.257 privileged against
0.339/0.341/0.341 control — distributions that do not touch, ~15 pooled sd apart.

The reading: the 25-dim teacher's extra action rate is **system identification**. It
has to wiggle and watch the response to infer effective inertia and gain from the
same 25 numbers it acts on. Hand it the draw and the probing motion disappears at no
cost to righting. That lands directly on `m_sm` and `m_time` — the two quantities
the penalty budget exists to buy down — so the block pays for a quarter of the
budget before tuning starts.

### The same control at the tuned weights

Repeated at the shipped tail weights rather than at zero penalty, 3 seeds, evaluated
on a held-out eval seed over 1000 drops:

| obs | seeds | mean | sd |
|---|---|---:|---:|
| 73-dim privileged | 51.6 / 54.5 / 55.2 | **53.8%** | **1.9** |
| 25-dim control | 54.3 / 44.9 / 41.5 | 46.9% | 6.6 |

**+6.9 pp, SE 4.0, t = 1.72 — not significant at n = 3.** The privileged teacher is
nominally better under the budget but this does not establish it; resolving 7 pp
against the control's spread needs ~8 seeds per arm (~1.5 h).

What the two arms *do* show consistently is **variance**: control seed sd is 6.6 here
and 4.4 penalty-free, against 1.9 and 1.8 privileged. The 25-dim teacher sometimes
finds a good policy under a tight budget and sometimes does not (54.3% vs 41.5% on
the same weights); the privileged one lands in the same place every time. Both arms
reach the same smoothness (`m_sm` 0.091 vs 0.093) — the budget forces that — so the
difference is in how reliably each converts a fixed motion budget into righting.

Note the tuned-weight control mean (46.9%) is *below* its own penalty-free baseline
(49.7%), while the privileged teacher gains 6.5 pp from the same budget. That is
suggestive of the penalty budget being harder to pay when motion is also doing
system identification, but at n = 3 it is a hypothesis, not a result.

## Baselines (all penalties 0, 3 seeds)

| metric | tail | no-tail |
|---|---:|---:|
| success (500 drops) | **47.3%** ±1.8 | **30.4%** ±3.5 |
| median tilt f/r | 26 / 26 deg | 35 / 33 deg |
| task reward/step | 1.665 (`r_pos` 1.448 + `r_bonus` 0.217) | 1.372 (1.246 + 0.126) |
| `m_sm` action rate | 0.250 | 0.200 |
| `m_en` torque | 0.464 | 0.184 |
| `m_av` body angular velocity | 18.3 | 20.4 |
| `m_jv` joint velocity | 123 | 130 |
| `m_time` simultaneity | 0.353 | 0.304 |
| \|action\| roll/pitch/tail | 0.72 / 0.66 / 0.76 | 0.73 / 0.78 / 0.65 |
| joint travel rot1/pitch/rot2/tail | 8.5 / 3.5 / 8.3 / 3.1 rad | 8.8 / 3.0 / 8.9 / 0.0 rad |
| final \|omega\| | 5.04 rad/s | 5.44 rad/s |

Both robots **end the drop still rotating at ~5 rad/s** and spend **8.5 rad of roll
travel** in a 0.74 s fall — far more motion than the maneuver needs. That is what the
penalties are for.

## Per-term sweeps

Each term swept alone, other four at zero, seed 0. `m` is the quantity that term
prices; `%` is the grid position (fraction of that variant's baseline task reward).

### Tail

| term | 2% | 6% | 18% | 50% | knee |
|---|---|---|---|---|---|
| `w_sm` | 49.2% / −9% | 46.2% / −12% | **50.0% / −43%** | 43.0% / −63% | 50% |
| `w_en` | 41.2% / −0% | 49.2% / +1% | 49.8% / −1% | **52.2% / −11%** | none |
| `w_av` | 45.0% / +6% | 44.4% / −1% | 46.8% / −14% | 47.2% / −29% | none |
| `w_jv` | 43.8% / +6% | 48.4% / −1% | 44.6% / −12% | 33.8% / −47% | **50%** |
| `w_time` | 47.8% / −11% | 45.2% / −30% | **48.4% / −58%** | 28.2% / −68% | **50%** |

- **`w_time` at 18% is the best single lever, and it is not close.** Simultaneity
  −58%, but also action rate −55% (0.353 → 0.113), joint velocity −14%, torque −13%,
  at 48.4% success. One weight, four magnitudes down, free.
- **`w_sm` at 18% is second and largely redundant with it** — 50.0%, `m_sm` −43%,
  and it pulls `m_time` down 46% by itself.
- **`w_av` is the weak term and is actively counterproductive.** At 18% and 50% the
  magnitudes it does *not* price go up: `m_sm` 0.287 and 0.323 against a 0.250
  baseline (+15%, +29%), `m_time` 0.425 and 0.459. Contact is disabled, angular
  momentum is conserved, and the policy cannot shed rotation — it only thrashes
  trying. Spending budget here makes the robot move *more*.
- **`w_en` has no knee and barely compresses** (50% of budget buys −11%). Expected:
  the Teensy floors any out-of-deadband command at `minPWM`, so torque is not
  continuously trimmable.

### No-tail

| term | 2% | 6% | 18% | 50% | knee |
|---|---|---|---|---|---|
| `w_sm` | 25.8% / −10% | 26.8% / −20% | 27.6% / −47% | **5.6% / collapse** | **50%** |
| `w_en` | 27.0% / −9% | 27.8% / −11% | 24.2% / −16% | 16.6% / −47% | 50% |
| `w_av` | 26.0% / −4% | 31.2% / −3% | 27.6% / −19% | 24.0% / −28% | none |
| `w_jv` | 28.0% / −2% | 28.0% / −11% | 26.2% / −27% | 23.4% / −46% | none |
| `w_time` | 27.8% / −2% | 26.6% / −25% | 24.6% / −54% | **6.6% / collapse** | **50%** |

**No grid point beats the 30.4% baseline.** The best cell — `w_av` at 6%, 31.2% —
moved its own magnitude by −3%, i.e. it did nothing and drifted up inside the noise
floor. Meanwhile `w_sm` and `w_time` at 50% collapse outright: 5.6% and 6.6% success
at 91° and 75° tilt, with `m_av` 0.42 and `m_jv` 6.3 against baselines of 20 and 130.
That is a robot holding still while it falls.

So the two variants are not the same curve at different scales. Tail *gains* from
`w_time`/`w_sm` at 18% and tolerates `w_en` at 50%; no-tail pays for every penalty
and detonates at 50% on the two levers tail likes best.

## Combining the terms

Penalties add. Nominal budget is the sum of the grid fractions.

### Tail — the budget buys success

| combo | budget | success | tilt | `m_sm` | `m_en` | `m_av` | `m_jv` | `m_time` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0% | 47.3% | 31 | 0.250 | 0.464 | 18.3 | 123 | 0.353 |
| `t_mild` | 24% | 47.2% | 29 | 0.186 | 0.423 | 18.4 | 113 | 0.239 |
| `t_lean` | 36% | 50.1% | 30 | 0.083 | 0.370 | 17.4 | 104 | 0.119 |
| **`t_cool`** | **54%** | **55.3%** | **28** | **0.091** | **0.359** | **16.1** | **86.6** | **0.129** |
| `t_noav` | 60% | 41.2% | 34 | 0.070 | 0.338 | 12.2 | 60.8 | 0.103 |
| `t_bal` | 72% | 41.8% | 34 | 0.080 | 0.353 | 11.4 | 58.4 | 0.119 |
| `t_enhot` | 86% | 41.0% | 35 | 0.061 | 0.318 | 13.4 | 71.6 | 0.086 |

*(`t_cool` and `t_lean` are 3-seed means; the rest are seed 0.)*

**`t_cool` is strictly better than penalty-free**: 55.3% ±1.8 against 47.3% ±1.8,
**+8.0 pp at 5.4 standard errors**, while cutting action rate 64%, torque 23%,
angular velocity 12%, joint velocity 29% and simultaneity 63%. Final |omega| falls
5.04 → 4.00 rad/s and median tilt 26/26 → 24/21 deg. This is the one place in the
study where the budget buys success rather than costing it.

The ceiling is between 54% and 60%: every combo at 60% and above lands at ~41%.
`t_lean` (+2.8 pp ±2.2) is inside noise, so the three extra terms in `t_cool` do real
work — visible as `m_jv` 86.6 vs 104 and `m_av` 16.1 vs 17.4.

### No-tail — the budget buys smoothness, at a cost that stays inside noise

| combo | budget | success | tilt | `m_sm` | `m_en` | `m_av` | `m_jv` | `m_time` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0% | 30.4% | 40 | 0.200 | 0.184 | 20.4 | 130 | 0.304 |
| `n_micro` | 8% | 24.8% | 43 | 0.169 | 0.150 | 19.1 | 114 | 0.211 |
| `n_lean` | 12% | 26.3% | 43 | 0.129 | 0.166 | 19.2 | 115 | 0.183 |
| **`n_mild`** | **24%** | **26.6%** | 43 | **0.129** | **0.148** | 18.4 | **99.3** | **0.168** |
| `n_jvmix` | 36% | 23.4% | 45 | 0.131 | 0.135 | 16.5 | 78.4 | 0.171 |
| `n_lean18` | 36% | **6.8%** | **86** | 0.055 | 0.096 | 1.74 | 22.1 | 0.096 |
| `n_cool` | 54% | **6.6%** | **91** | 0.011 | 0.014 | 0.20 | 6.37 | 0.015 |

*(`n_mild` and `n_lean` are 3-seed means; the rest are seed 0.)*

**Nothing beats the penalty-free baseline.** `n_mild` and `n_lean` sit ~4 pp below it,
which at these sds is not significant (t = 1.6 and 2.0) but is not evidence of
improvement either. The honest framing is a **ceiling search**: `n_mild` buys −36%
action rate, −45% simultaneity, −24% joint velocity and −19% torque for a success
cost inside the noise floor, and it is the better of the two because at equal success
it compresses `m_jv` and `m_en` roughly twice as hard.

The ceiling is between 24% and 36%, and it is a cliff, not a slope: `n_lean18` and
`n_cool` lose two thirds of their success and land at 86–91° tilt. Note that
`n_jvmix` and `n_lean18` are the *same* 36% budget with different shapes — 23.4% vs
6.8%. Past the ceiling, shape matters enormously; below it, only the total does.

## Shipped weights

Per variant — a single scale factor cannot express this, which is why
`notail_penalty_scale` was removed:

```
tail     w_sm = 0.399   w_en = 0.646   w_av = 0.00547   w_jv = 0.000815   w_time = 0.848
no-tail  w_sm = 0.412   w_en = 0.447   w_av = 0.0       w_jv = 0.000633   w_time = 0.271
```

The no-tail/tail ratios are `sm` 1.03, `en` 0.69, `av` 0.00, `jv` 0.78, `time` 0.32.
`w_sm` is essentially identical across variants while `w_time` differs 3x and `w_av`
is dropped entirely — no scalar produces that.

If you change one, change another to compensate: the **budget** is the constraint,
not any single weight. Watch `m_*` in `docs/evaluate.py --stats`, not the reward. The
tell for the passive collapse is every magnitude dropping together while median tilt
climbs past ~45 deg.

## Rejected: penalizing command oscillation

Retained from the previous study — the mechanism is physical and does not depend on
the weights or the observation, and nothing here contradicts it. Three terms were
built to suppress the visible oscillation in the commanded joint targets (`w_osc` on
the raw action, `w_rev` on reversal rate, the same two on the post-filter command,
and `w_track` on tracking error). **All were removed.**

Quadratic terms are gamed by shrinking: `m_sm`, `m_osc` and any squared quantity fall
4x when command amplitude halves while the reversal pattern is bit-identical, so the
policy shrinks rather than straightens. But the real reason is physical — sorting 200
drops by command reversal rate, the **most**-reversing quartile succeeded 75.0% against
54.9% for the least. A zero-angular-momentum reorientation takes its net rotation from
the geometric phase of a *closed loop* in shape space: bend, twist, unbend,
counter-twist. A monotone shape change nets zero rotation. The cycling is the
mechanism, not a pathology, and every one of those penalties was taxing it.

The lesson for the next term: **check whether the behavior correlates with success
before designing a penalty against it.** The amplitude-side terms that survive
(`w_sm`, `w_time`) are fine precisely because they constrain how *far* the command
moves, not whether it is allowed to turn around.

## Reproducing

`docs/sweep.py` takes a JSON list of jobs — `{name, weights, seed, variant, steps,
privileged}` — trains each with the weights exported as `CAT_W_*`, evaluates it, and
appends one JSONL record per run. `docs/sweep_jobs/gen_jobs.py` builds the reward-unit grid
from measured baselines.

```bash
# 1. baselines (all penalties 0) -> measures m_* and the seed noise floor
python docs/sweep.py docs/sweep_jobs/jobs_p1.json --out docs/reward_tuning_runs.jsonl \
    --parallel 3 --steps 1000000 --envs 10 --gradient-steps 1 --episodes 500 \
    --model-dir sweep_models --log-dir sweep_logs

# 2. size the per-term grid from those baselines, then run it
python docs/sweep_jobs/gen_jobs.py docs/reward_tuning_runs.jsonl --out docs/sweep_jobs/jobs_p2.json
python docs/sweep.py docs/sweep_jobs/jobs_p2.json --out docs/reward_tuning_runs.jsonl ...
```

A single config, without the harness:

```bash
CAT_W_TIME=0.848 python train.py --variant tail --steps 1000000 --out t.zip
python docs/evaluate.py --agent teacher --policy t.zip --episodes 500 --stats
```

`--no-privileged` on `train.py` trains the 25-dim control arm; `docs/evaluate.py`
reads the observation width back off the checkpoint, so pre-DR teachers evaluate
with no flag.

**Throughput matters for planning.** On a 10-core M4 Pro, aggregate throughput
saturates at **3 concurrent runs** (3326 steps/s; 4-way gives 3238, 5-way 3141) —
more parallelism is net negative. One 1M-step run is ~8 min solo, ~15 min at
`--parallel 3`. The full study was **77 runs / 77M steps ≈ 7.5 h wall clock**.
