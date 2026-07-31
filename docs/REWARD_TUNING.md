# Reward-weight tuning

How the five penalty weights in `cat_env/cat_env.py` were set, and the sweep that
set them. Run with `docs/sweep.py`; every number below is 500 random drops of the
privileged teacher via `docs/evaluate.py`.

## Method

**Start from zero.** All five penalties (`w_sm`, `w_en`, `w_av`, `w_jv`, `w_time`)
were set to 0, leaving only the task terms `w_pos` and `w_bonus`. That baseline is
the reference: the best success rate the task reward alone can reach, and the
magnitude of each penalized quantity when nothing discourages it.

**Raise one weight at a time until success drops.** Each term was swept alone, with
the other four at zero, over a 4-point geometric grid, then combined.

**Size the grid in reward units, not raw weights.** A raw weight is meaningless on
its own: `mean(omega^2)` ~ 15 and `mean(dAction^2)` ~ 0.24, so `w = 0.1` is crushing
for one and negligible for the other. Each grid point is instead chosen so that
`w * m_baseline` equals a fixed **fraction of the baseline task reward** (1.70/step):
2%, 6%, 18%, 50%. That makes a given grid position mean the same thing for every
term and makes the knees comparable.

**Read the change in the term, not its absolute value.** The env reports each
penalty's *unweighted* magnitude as `m_*` in `info` beside the weighted `r_*`
(`docs/evaluate.py --stats`). This is the signal to tune on, for two reasons:

- `r_*` is zero at `w = 0` and confounds weight with behavior everywhere else, so it
  cannot answer "did this weight change what the robot does?". `m_*` can.
- success rate is noisy (see below) while `m_*` responds directly and with far less
  variance, so the magnitude column identifies a working weight several grid points
  before the success column can confirm it is safe.

**Noise floor.** Two grid points whose magnitude did not move (`w_sm` and `w_en` at
2%) scored 52.0% and 47.4% against a 57.8% baseline. Since nothing was meaningfully
penalized in either, that spread is run-to-run training variance: about ±5 pp, well
above the ±2.2 pp sampling error of a 500-drop evaluation. A success drop is
therefore only treated as real when it exceeds ~8 pp **and** the magnitude has moved
**and** median tilt rises with it. The passive-collapse failure mode satisfies all
three at once and is unmistakable.

## Baseline (all penalties 0)

| metric | value |
|---|---|
| success (500 drops) | 57.8% |
| median tilt f/r | 26 / 27 deg |
| task reward/step | 1.70 (`r_pos` 1.44 + `r_bonus` 0.26) |
| `m_sm` action rate | 0.237 |
| `m_en` torque | 0.237 |
| `m_av` body angular velocity | 15.25 |
| `m_jv` joint velocity | 110.2 |
| `m_time` simultaneity | 0.314 |
| \|action\| roll/pitch/tail | 0.76 / 0.66 / 0.76 |
| joint travel rot1/pitch/rot2/tail | 8.2 / 2.7 / 8.1 / 2.9 rad |
| final \|omega\| | 4.11 rad/s |

Two of these are the reason the new penalties exist. The robot **ends the drop still
rotating at 4.1 rad/s** — it reaches upright but does not settle there — and it
spends **8.2 rad of roll travel** in a 0.74 s fall, far more motion than the
maneuver needs.

## Training configuration

Checked first, since an under-trained policy would make every weight look harmless:

| config | steps | grad steps / env step | success | wall |
|---|---:|---:|---:|---:|
| 10 envs, `gradient_steps=1` | 1M | 0.1 | **57.8%** | 574 s |
| 5 envs, `gradient_steps=5` | 500k | 1.0 | 50.4% | 1592 s |

Raising the update-to-data ratio to the SAC default of 1.0 did not help and cost 3x
the wall clock. Not a clean comparison — the faster config also sees twice the
environment data — but it settles the practical question, so the sweep runs on the
1M-step, `gradient_steps=1` config.

## Per-term sweeps

Each term swept alone, other four at zero. `m` is the quantity that term prices;
the % column is the grid position (fraction of baseline task reward).

### `w_sm` — action rate

| w_sm | % | success | m_sm | tilt |
|---:|---:|---:|---:|---:|
| 0 | – | 57.8% | 0.237 | 27 |
| 0.143 | 2 | 52.0% | 0.233 | 29 |
| 0.430 | 6 | 57.4% | 0.177 | 27 |
| **1.289** | **18** | **57.8%** | **0.120** | **26** |
| 3.581 | 50 | 38.6% | 0.065 | 39 |

Knee at 50%. Half the action rate for free at 18%.

### `w_en` — applied torque

| w_en | % | success | m_en | tilt |
|---:|---:|---:|---:|---:|
| 0 | – | 57.8% | 0.237 | 27 |
| 0.143 | 2 | 47.4% | 0.225 | 29 |
| 0.430 | 6 | 57.0% | 0.226 | 26 |
| 1.290 | 18 | 52.2% | 0.202 | 28 |
| **3.584** | **50** | **63.8%** | **0.137** | **24** |

**No knee inside the grid** — the strongest setting scored best in the entire
single-term sweep. Torque is also the least compressible quantity (18% of task
reward buys only −15%), which fits the hardware model: the Teensy floors any
out-of-deadband command at `minPWM`, so torque cannot be trimmed continuously.

### `w_av` — body angular velocity

| w_av | % | success | m_av | tilt |
|---:|---:|---:|---:|---:|
| 0 | – | 57.8% | 15.25 | 27 |
| 0.0022 | 2 | 58.4% | 14.71 | 25 |
| 0.0067 | 6 | 53.2% | 14.69 | 28 |
| 0.0201 | 18 | 51.4% | 12.81 | 29 |
| 0.0557 | 50 | 34.0% | 6.57 | 43 |

**The weakest term, and the only one with poor value alone.** 18% of task reward
buys only −16% omega. The reason is physical: contact is disabled, so angular
momentum is conserved and the policy cannot *remove* rotation, only move it between
the bodies and the joints. Worse, at 6–18% the terms it does not price go **up**
(`m_sm` 0.294, `m_time` 0.402 at 6%) — the policy thrashes trying to shed rotation
it cannot shed. Kept at 6% in the final weights, where it is nearly free.

### `w_jv` — joint velocity

| w_jv | % | success | m_jv | tilt |
|---:|---:|---:|---:|---:|
| 0 | – | 57.8% | 110.2 | 27 |
| 0.00031 | 2 | 55.8% | 117.7 | 26 |
| 0.00093 | 6 | 50.4% | 117.6 | 28 |
| 0.00278 | 18 | 54.2% | 82.6 | 27 |
| **0.00771** | **50** | 52.0% | **54.6** | 28 |

**No knee inside the grid.** Halves joint velocity and pulls every other magnitude
down with it. Note `m_jv` has the noisiest baseline of the five (sd 20 across 22
runs), so only the 18% and 50% points are outside run-to-run spread.

### `w_time` — simultaneous multi-joint motion

| w_time | % | success | m_time | m_sm | final \|omega\| | tilt |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | – | 57.8% | 0.314 | 0.237 | 4.11 | 27 |
| 0.108 | 2 | 55.4% | 0.294 | 0.217 | 3.85 | 27 |
| 0.325 | 6 | 56.4% | 0.238 | 0.168 | 3.52 | 26 |
| **0.976** | **18** | **58.8%** | **0.129** | **0.092** | **2.54** | **26** |
| 2.710 | 50 | 32.4% | 0.073 | 0.048 | 3.34 | 42 |

**The best single lever.** At 18% it cuts simultaneity 59%, action rate 61%, and
final angular velocity 38% — the lowest final |omega| of any run in the study,
*better than the term that penalizes omega directly*. Asking the joints to move one
at a time produces a controlled, sequential maneuver that settles; penalizing
rotation directly just makes the policy fight conservation of angular momentum.

## Combining the terms

Penalties add. Five terms at their individual knees total 142% of the task reward,
and that is far past the passive attractor even though every term was harmless alone:

| combo | nominal cost | success | tilt | m_sm | m_en | m_av | m_jv | m_time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0% | 57.8% | 27 | 0.237 | 0.237 | 15.25 | 110.2 | 0.314 |
| **cool** | **54%** | **55.0%** | **29** | **0.105** | **0.149** | **9.19** | **55.5** | **0.150** |
| noav | 136% | 23.2% | 51 | 0.037 | 0.039 | 6.53 | 22.7 | 0.050 |
| knee | 142% | 9.4% | 80 | 0.008 | 0.015 | 0.55 | 6.67 | 0.015 |
| hot | 268% | 11.0% | 62 | 0.006 | 0.011 | 2.98 | 13.4 | 0.010 |

What matters is the **total** penalty budget, not which term carries it: dropping
`w_av` entirely (`noav`, 136%) collapsed just like `knee` at 142%. The usable budget
is around 50% of task reward, and `combo_cool` spends it to halve *every* penalized
quantity at once for a success change inside the noise floor.

Re-spending that same ~50% budget on the strongest single levers made things worse,
not better — the terms are not independent, and a balanced spread beats concentration:

| combo | budget | success | tilt |
|---|---:|---:|---:|
| `cool` (6/18/6/18/6 %) | 54% | **55.0 / 62.8 / 50.4%** (seeds 0/1/2) | 29 / 25 / 30 |
| `v4` — same shape, milder | 38% | 62.0% | 26 |
| `v3` — av dropped onto time | 54% | 51.4% | 30 |
| `v2` — shifted onto time+jv | 50% | 44.0% | 34 |

`combo_cool` over three seeds averages **56.1% (sd 6.2)** against a 57.8% baseline:
success is statistically unchanged while action rate falls 62%, torque 46%, joint
velocity 45%, simultaneity 60% and body angular velocity 31%. That is the trade the
penalties exist to make, and it is the shipped setting.

## The no-tail variant needs a smaller budget

At the tailed weights the no-tail robot collapses outright. Walking the whole budget
down by a single scale factor:

| scale | budget | success | tilt | m_sm | m_time |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 54% | 6.2% | 91 | 0.045 | 0.078 |
| 0.70 | 38% | 17.4% | 57 | 0.103 | 0.143 |
| 0.50 | 27% | 28.2% | 43 | 0.180 | 0.253 |
| **0.25** | **14%** | **34.0%** | **38** | **0.176** | **0.261** |
| 0.125 | 7% | 26.6% | 42 | 0.249 | 0.360 |
| 0.00 | 0% | 29.6% | 40 | 0.242 | 0.333 |

This is the same effect `docs/EVALUATION.md` found for `w_sm` alone, and the reason
`notail_penalty_scale` exists: **a variant with less authority reaches the passive
optimum at a lower penalty budget.** The bottom four rows are all within the ±5 pp
noise floor of each other, so the honest statement is that no-tail tolerates *some*
penalty and peaks around a quarter of the tailed budget — not that 14% is precisely
optimal. It is still worth applying: at 0.25 the no-tail policy is measurably
smoother than penalty-free (`m_sm` −27%, `m_time` −22%) at no cost to success.

## Shipped weights

```
w_sm = 0.43     w_en = 1.29     w_av = 0.0067     w_jv = 0.0028     w_time = 0.33
notail_penalty_scale = 0.25
```

If you change one, change another to compensate — the budget is the constraint.
Watch `m_*` in `docs/evaluate.py --stats`, not the reward: the tell for the passive
collapse is every magnitude dropping together while median tilt climbs past ~35 deg.

## Rejected: penalizing command oscillation

The commanded joint targets visibly oscillate (see `plots/plot_joint_tracking.py`),
which looks like a sim2real liability. Three terms were built and swept to suppress
it. **All three were removed.** In order:

| term | definition | result |
|---|---|---|
| `w_osc` on the raw action | `mean((aₜ − 2aₜ₋₁ + aₜ₋₂)²)` | at 18%, `m_osc` −56% but the reversal rate did not move (0.393 → 0.400); −26 pp success |
| `w_rev` on the raw action | fraction of channels reversing direction | unmoved at **every** weight up to 50% of task reward (0.393 → 0.424) |
| `w_osc` / `w_rev` on the executed command | same, on the post-filter command | at 18% `w_osc` **collapsed to 10.0%** success, 74° tilt, 1.66 rad roll travel |
| `w_track` | `mean((joint target − q)²)` | worked as designed (−39% at 18%, reversal rate preserved) but cost ~10 pp success and was never shown affordable budget-neutral |

Two things went wrong, and both are worth remembering:

**1. Quadratic terms are gamed by shrinking.** `m_sm`, `m_osc` and any other squared
quantity all fall 4x when the command amplitude halves, while the reversal pattern is
bit-identical. Shrinking is always cheaper than straightening, so the policy shrinks.

**2. The statistic was measuring the sampled action, not the policy.** SAC's
exploration noise dominates step-to-step differences: in training the raw action's
reversal fraction is **0.564, above the 0.500 of pure white noise**. A per-step
smoothness statistic taken there mostly prices the entropy-tuned sigma. (This is also
true of `m_sm` as shipped, and is a plausible second reason high `w_sm` collapses
training — it fights the entropy objective, not only the passive attractor.)

**But the real reason is physical, and it invalidates the whole idea.** Sorting 200
drops by how much the command reverses:

| | reversal rate | success |
|---|---:|---:|
| successful drops | 0.256 | – |
| failed drops | 0.237 | – |
| fewest reversals (bottom quartile) | ≤0.19 | **54.9%** |
| most reversals (top quartile) | ≥0.30 | **75.0%** |

**More oscillation predicts more success.** A zero-angular-momentum reorientation
takes its net rotation from the geometric phase of a *closed loop* in shape space —
bend, twist, unbend, counter-twist. A monotone shape change nets zero rotation, so
the cycling is not a pathology, it is the entire mechanism. Every penalty above was
taxing the maneuver; the one that "worked best" simply stopped the robot righting
itself.

The lesson for the next term: **check whether the behavior correlates with success
before designing a penalty against it.** The amplitude-side terms that survive
(`w_sm`, `w_time`) are fine precisely because they constrain how *far* the command
moves, not whether it is allowed to turn around.

## Reproducing

`docs/sweep.py` takes a JSON list of jobs — `{name, weights, seed, variant, steps}` —
trains each with the weights exported as `CAT_W_*`, evaluates it, and appends one
JSONL record per run:

```bash
cat > jobs.json <<'EOF'
[{"name": "baseline",  "weights": {"sm":0,"en":0,"av":0,"jv":0,"time":0}, "seed": 0},
 {"name": "wtime_0.18","weights": {"sm":0,"en":0,"av":0,"jv":0,"time":0.976}, "seed": 0}]
EOF
python docs/sweep.py jobs.json --out results.jsonl --parallel 3 \
    --steps 1000000 --envs 10 --gradient-steps 1 --episodes 500
```

A single config, without the sweep harness:

```bash
CAT_W_TIME=0.976 python train.py --variant tail --steps 1000000 --out t.zip
python docs/evaluate.py --agent teacher --policy t.zip --episodes 500 --stats
```

Wall clock on a 10-core M-series: ~10 min per 1M-step run solo, ~17 min with three
in parallel. The full study was **37 training runs, 36.5M environment steps**; every
record is archived in `docs/reward_tuning_runs.jsonl` (one JSON object per run, with
the weights, success rate, tilts and all five magnitudes), so every table above can
be re-derived without retraining.

