#!/usr/bin/env python3
"""Normalize the final-experiment telemetry layout without deleting source data.

The script is intentionally idempotent. It moves recognized final-experiment trials
into a canonical directory tree, moves old/ambiguous material into ``archive/``, and
writes a SHA-256 provenance manifest recording every move.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = ROOT / "telemetry"
RAW = TELEMETRY / "raw"
ARCHIVE = TELEMETRY / "archive"
MANIFEST = TELEMETRY / "manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_path(
    morphology: str,
    sweep: str,
    roll_deg: int,
    pitch_deg: int,
    date: str,
    rep: int,
) -> Path:
    morph_dir = "with_tail" if morphology == "WT" else "no_tail"
    if sweep == "roll":
        condition = f"roll_{roll_deg:03d}deg"
        filename = f"{morphology.lower()}_roll_{roll_deg:03d}deg_{date}_rep{rep:02d}.csv"
    else:
        condition = f"roll_{roll_deg:03d}deg_pitch_{pitch_deg:+04d}deg"
        filename = (
            f"{morphology.lower()}_roll_{roll_deg:03d}deg_"
            f"pitch_{pitch_deg:+04d}deg_{date}_rep{rep:02d}.csv"
        )
    return RAW / morph_dir / sweep / condition / filename


def trial_moves() -> list[tuple[Path, Path, str]]:
    moves: list[tuple[Path, Path, str]] = []

    def add(source: Path, destination: Path, note: str = "") -> None:
        moves.append((source, destination, note))

    # With-tail roll-only trials.
    for folder, date in (("WT_0810", "0810"), ("WT_0816", "0816")):
        base = TELEMETRY / folder
        if base.exists():
            for source in sorted(base.glob("*.csv")):
                match = re.fullmatch(r"(45|90|180)deg(?:_WTsuccess)?_\d{4}_(\d+)\.csv", source.name)
                if match:
                    angle, rep = map(int, match.groups())
                    add(source, canonical_path("WT", "roll", angle, 0, date, rep))

    # No-tail roll-only trials. The 0810/180 files were mislabeled locally;
    # content hashes match the reference branch's 0817 trials.
    base = TELEMETRY / "NT_0810"
    if base.exists():
        for source in sorted(base.glob("*.csv")):
            match = re.fullmatch(r"(45|90|180)deg_NTsuccess_0810_(\d+)\.csv", source.name)
            if match:
                angle, rep = map(int, match.groups())
                date = "0817" if angle == 180 else "0810"
                note = "Corrected date from 0810 to 0817 by SHA-256 match to reference branch." if angle == 180 else ""
                add(source, canonical_path("NT", "roll", angle, 0, date, rep), note)

    for angle in (45, 90):
        for source in sorted(TELEMETRY.glob(f"NT{angle}_0816_*.csv")):
            rep = int(source.stem.rsplit("_", 1)[1])
            add(source, canonical_path("NT", "roll", angle, 0, "0816", rep))

    for source in (TELEMETRY / "NT180_0816_1.csv", TELEMETRY / "NT180_0816_2.csv"):
        if source.exists():
            rep = int(source.stem.rsplit("_", 1)[1])
            add(source, canonical_path("NT", "roll", 180, 0, "0816", rep))
    source = TELEMETRY / "NT180_0817_3.csv"
    if source.exists():
        add(
            source,
            canonical_path("NT", "roll", 180, 0, "0816", 3),
            "Corrected date from 0817 to 0816 by SHA-256 match to reference branch.",
        )

    # With-tail roll-180 + pitch trials.
    base = TELEMETRY / "WT_Pitch_0816"
    if base.exists():
        for source in sorted(base.glob("P*.csv")):
            match = re.fullmatch(r"P(15|30)_0816_(\d+)\.csv", source.name)
            if match:
                pitch, rep = map(int, match.groups())
                add(source, canonical_path("WT", "roll_pitch", 180, pitch, "0816", rep))
    for source in sorted(TELEMETRY.glob("WTP45_0817_*.csv")):
        rep = int(source.stem.rsplit("_", 1)[1])
        if rep == 7:
            add(source, ARCHIVE / "excluded" / "spare" / "wt_roll180_pitch045_0817_rep07_spare.csv", "Pre-designated spare trial.")
        else:
            add(source, canonical_path("WT", "roll_pitch", 180, 45, "0817", rep))

    # No-tail roll-180 + pitch trials.
    special = TELEMETRY / "NT P15_0817_1.csv"
    if special.exists():
        add(special, canonical_path("NT", "roll_pitch", 180, 15, "0817", 1), "Normalized embedded space in original filename.")
    for pitch in (15, 30, 45):
        for source in sorted(TELEMETRY.glob(f"NTP{pitch}_0817_*.csv")):
            if "(?)" in source.name:
                add(
                    source,
                    ARCHIVE / "excluded" / "ambiguous" / "nt_roll180_pitch015_0817_rep06_alternate.csv",
                    "Ambiguous alternate for rep 6; excluded from analysis.",
                )
                continue
            rep = int(source.stem.rsplit("_", 1)[1])
            add(source, canonical_path("NT", "roll_pitch", 180, pitch, "0817", rep))

    # Known but non-final or insufficiently identified files are retained in archive.
    ambiguous = ("NT180_0813_3.csv", "P-15_0816_1.csv")
    for name in ambiguous:
        source = TELEMETRY / name
        if source.exists():
            add(source, ARCHIVE / "excluded" / "ambiguous" / name.replace(" ", "_"), "Insufficient metadata for final analysis.")

    return moves


def auxiliary_moves() -> list[tuple[Path, Path, str]]:
    moves: list[tuple[Path, Path, str]] = []
    duplicate = TELEMETRY / "telemetry"
    if duplicate.exists():
        moves.append((duplicate, ARCHIVE / "duplicate_snapshot", "Nested snapshot containing duplicate files."))
    old_archive = TELEMETRY / "_archive"
    if old_archive.exists():
        moves.append((old_archive, ARCHIVE / "preliminary" / "old_archive", "Legacy preliminary trials."))

    preliminary_names = (
        "180deg_1.csv", "180deg_fail1.csv", "180deg_fail2.csv", "180deg_success?.csv",
        "45deg_fail1.csv", "90deg_1.csv", "90deg_fail1.csv", "90deg_fail2.csv", "90deg_success?.csv",
    )
    for name in preliminary_names:
        source = TELEMETRY / name
        if source.exists():
            clean_name = name.replace("?", "_uncertain")
            moves.append((source, ARCHIVE / "preliminary" / clean_name, "Legacy success/failure-labeled trial."))

    for source in sorted(TELEMETRY.glob("telemetry_*.csv")):
        moves.append((source, ARCHIVE / "unclassified_captures" / source.name, "Timestamp-only capture; not a labeled final trial."))

    for source in sorted(TELEMETRY.glob("*.png")):
        moves.append((source, TELEMETRY / "derived" / "legacy_sim2real" / source.name, "Legacy generated figure."))
    notebook = TELEMETRY / "sim2real.ipynb"
    if notebook.exists():
        moves.append((notebook, ARCHIVE / "legacy_analysis" / notebook.name, "Legacy exploratory notebook."))
    return moves


def main() -> None:
    rows_by_path: dict[str, dict[str, str]] = {}
    if MANIFEST.exists():
        with MANIFEST.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_path[row["canonical_path"]] = row

    for source, destination, note in trial_moves() + auxiliary_moves():
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        digest = sha256(source) if source.is_file() else ""
        original = source.relative_to(ROOT).as_posix()
        shutil.move(str(source), str(destination))
        canonical = destination.relative_to(ROOT).as_posix()
        rows_by_path[canonical] = {
            "original_path": original,
            "canonical_path": canonical,
            "sha256": digest,
            "status": "analysis_input" if destination.is_relative_to(RAW) else "archived",
            "note": note,
        }

    # Include canonical files on repeat runs so the manifest remains complete.
    known = set(rows_by_path)
    for path in sorted(RAW.rglob("*.csv")) if RAW.exists() else []:
        relative = path.relative_to(ROOT).as_posix()
        if relative not in known:
            rows_by_path[relative] = {
                "original_path": "",
                "canonical_path": relative,
                "sha256": sha256(path),
                "status": "analysis_input",
                "note": "Already organized before this run.",
            }

    rows = sorted(rows_by_path.values(), key=lambda row: row["canonical_path"])
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("original_path", "canonical_path", "sha256", "status", "note"))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest entries: {len(rows)}")
    print(f"Analysis inputs: {sum(r['status'] == 'analysis_input' for r in rows)} CSV files")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
