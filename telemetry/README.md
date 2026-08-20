# Telemetry layout

- `raw/`: the 73 labeled final-experiment recordings used as candidate analysis inputs.
- `manifest.csv`: old-to-new paths, SHA-256 hashes, analysis/archive status, and metadata corrections.
- `archive/excluded/`: known spares and ambiguous trials excluded from analysis.
- `archive/unclassified_captures/`: timestamp-only captures lacking treatment/condition labels.
- `archive/preliminary/`: earlier success/failure-labeled trials.
- `archive/duplicate_snapshot/`: the pre-existing nested `telemetry/telemetry` snapshot.
- `derived/`: legacy generated figures. New statistical outputs live in `analysis/results/`.

Canonical codes:

- `wt`: with tail.
- `nt`: no tail.
- `roll_180deg_pitch_+015deg`: released near 180 degrees roll with +15 degrees initial pitch.

The former `NTP`/`NT P` files are therefore represented explicitly as
`nt_roll_180deg_pitch_...`; `WTP` files use the equivalent `wt_...` form.

Do not add files directly to `raw/` unless the filename includes morphology,
roll, pitch (when applicable), collection date, and replicate number. Preserve
excluded or uncertain data in `archive/` rather than deleting it.
