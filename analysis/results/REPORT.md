# Tail vs. no-tail telemetry analysis

## Bottom line

The coupled whole-cat score favors the tail by **+0.157** (95% CI +0.062 to +0.252, p=0.0016). This is **significant** at alpha=0.05.

The advantage is primarily better **front/rear alignment**, not clearly better whole-body uprightness. The result is exploratory because the metric was chosen after inspecting the data and collection date is partly confounded with morphology.

## Main results

Effects are adjusted differences: with tail minus no tail. Positive score effects favor the tail; negative angular-speed effects favor the tail.

| Outcome | Effect | 95% CI | HC3 p | Result |
|---|---:|---:|---:|---|
| Coupled whole-cat score | +0.157 | +0.062 to +0.252 | 0.0016 | Significant |
| Centerline uprightness | +0.021 | -0.024 to +0.067 | 0.3584 | Diagnostic: not significant |
| Front/rear alignment | +0.162 | +0.066 to +0.258 | 0.0014 | Diagnostic: significant |
| Angular speed (deg/s) | -13.7 | -162.7 to +135.4 | 0.8550 | Not significant |

The blocked permutation check agrees for the primary score (p=0.0008). Without date adjustment, its estimate is +0.112 (p=0.0017).

## Requested pooled comparisons

| Comparison | n | Effect | 95% CI | HC3 p | Holm p | Result |
|---|---:|---:|---:|---:|---:|---|
| Roll 180°, pitch 0/15/30/45° | 48 | +0.114 | -0.057 to +0.286 | 0.1841 | 0.1841 | Not significant |
| No pitch, roll 45/90/180° | 36 | +0.185 | +0.075 to +0.296 | 0.0019 | 0.0038 | Significant |

The pitch sweep is under-supported by condition/date overlap (only 18 usable overlap-stratum trials), so its uncertainty is large.

## Pitch leveling within the pitch sweep

This additional endpoint is the final absolute whole-cat fused pitch: 0° is level, so a negative tail effect favors the tail. The model adjusts for release-pitch condition, collection date, initial absolute whole-cat pitch, and initial angular speed.

| Outcome | n | Tail effect | 95% CI | HC3 p | Holm p across 3 requested tests | Result |
|---|---:|---:|---:|---:|---:|---|
| Absolute final whole-cat pitch | 48 | +0.82° | -8.43° to +10.08° | 0.8582 | 0.8582 | Not significant |

Signed pitch is not used as the primary leveling endpoint because positive and negative errors could cancel even when both are far from level.

## Time to righting

Righting is the first time the coupled score reaches 0.80 and remains there for 0.10 s. Trials that do not right by the common 2.5 m cutoff are assigned the 0.714 s horizon; this restricted-time definition avoids analyzing successful trials alone.

| Outcome | Tail effect | 95% CI | HC3 p | Holm p across 4 new endpoints |
|---|---:|---:|---:|---:|
| Restricted time to righting | -0.106 s | -0.169 to -0.043 | 0.0013 | 0.0040 |
| Mean posture deficit through cutoff | -0.059 | -0.095 to -0.023 | 0.0018 | 0.0040 |

The adjusted result estimates righting **106 ms sooner** with the tail. By the cutoff, 10/35 tail trials and 6/37 no-tail trials met the sustained threshold. Results remain significant at thresholds 0.75 (p=0.0005) and 0.85 (p=0.0064).

## Actuator effort—not measured energy

The telemetry logs target positions but not applied PWM, current, or voltage, so joules cannot be recovered. The available proxy reconstructs normalized PD duty from 50 Hz target error and joint motion. It omits the 1 kHz inner-loop waveform and must not be labeled electrical energy.

| Effort proxy | Tail effect | 95% CI | HC3 p | Holm p across 4 new endpoints |
|---|---:|---:|---:|---:|
| Three common spine actuators | -0.205 | -0.314 to -0.097 | 0.0004 | 0.0014 |
| All four actuators, including tail | +0.143 | +0.038 to +0.249 | 0.0085 | 0.0085 |

The tail reduces modeled effort in the three shared spine actuators but increases total modeled effort once its own actuator is included. Add voltage/current sensing and log applied PWM to answer the actual energy-consumption question.

## Metric

```text
center_up       = normalize(front_up + rear_up)
U               = 1 - whole_body_tilt / pi
A               = 1 - relative_front_rear_angle / pi
whole_cat_score = U * A
```

The score ranges from 0 to 1. It ignores common yaw, penalizes relative spine twist, and reaches 1 only when the whole-body centerline is upright and the front/rear frames are aligned. A level but bent pose therefore scores higher than a front-upright/rear-tilted pose with the same bend.

Whole-body fused roll/pitch and angular speed are retained as diagnostics. Literal joint-neutral straightness would require an additional calibrated joint-error measure because endpoint attitudes cannot detect internal joint cancellation.

## Analysis and limitations

- 72 drops analyzed at an exact 2.5 m fall-distance cutoff; 1 short drop was excluded.
- ANCOVA adjusts for release condition, collection date, initial pose score, and initial angular speed; confidence intervals use HC3 robust standard errors.
- Tail/no-tail assignment was not randomized or interleaved within collection session. Date and morphology are partly confounded, particularly at 15° and 30° pitch.
- The metric was developed after viewing these data. Treat all p-values as exploratory and test the frozen metric in a new randomized dataset.

## Power

Current 80% minimum detectable difference: **0.109 score units**.

Future balanced two-group design (two-sided alpha=0.05, 80% power):

| Difference to detect | Drops per morphology |
|---:|---:|
| 0.025 | 386 |
| 0.050 | 98 |
| 0.075 | 44 |
| 0.100 | 26 |

Randomize tail/no-tail within release condition and collection block. Divide each morphology total across conditions and round up.

See `SENSITIVITY.md` for cutoff sensitivity, `trial_metrics.csv` for trial-level values, and `time_effort_analysis.csv` for the new models. PNG files are retained only for statistically significant date-adjusted results.
