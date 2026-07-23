"""Discovery and loading of CWRU bearing vibration signals.

The CWRU archive is distributed as MATLAB v5 .mat files, one per run. Different
public mirrors (Kaggle, GitHub forks, the original Case Western site) name the
files differently: some keep the original numeric names (``97.mat``), others use
descriptive names (``IR007_1.mat``). Both are handled here.

Labels are used ONLY for evaluation. Training and thresholding see healthy data
exclusively -- see ``main.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

# --- Canonical CWRU 12 kHz drive-end file numbering -------------------------
# number -> (condition, fault_diameter_inches, motor_load_hp)
# Source: Case Western Reserve University Bearing Data Center, 12k Drive End
# Bearing Fault Data + Normal Baseline Data.
CWRU_FILE_MAP: dict[int, tuple[str, float, int]] = {}


def _register(numbers: tuple[int, ...], condition: str, size: float) -> None:
    for load, number in enumerate(numbers):  # loads are 0, 1, 2, 3 hp in order
        CWRU_FILE_MAP[number] = (condition, size, load)


_register((97, 98, 99, 100), "normal", 0.000)
_register((105, 106, 107, 108), "inner_race", 0.007)
_register((118, 119, 120, 121), "ball", 0.007)
_register((130, 131, 132, 133), "outer_race", 0.007)
_register((169, 170, 171, 172), "inner_race", 0.014)
_register((185, 186, 187, 188), "ball", 0.014)
_register((197, 198, 199, 200), "outer_race", 0.014)
_register((209, 210, 211, 212), "inner_race", 0.021)
_register((222, 223, 224, 225), "ball", 0.021)
_register((234, 235, 236, 237), "outer_race", 0.021)

_DESCRIPTIVE = re.compile(
    r"(?P<code>normal|nor|ir|b|or|ball|inner|outer)[\s_-]*"
    r"(?P<size>\d{3})?[\s_-]*(?:@(?P<clock>\d+))?[\s_-]*(?P<load>[0-3])?",
    re.IGNORECASE,
)

_CODE_TO_CONDITION = {
    "normal": "normal",
    "nor": "normal",
    "ir": "inner_race",
    "inner": "inner_race",
    "or": "outer_race",
    "outer": "outer_race",
    "b": "ball",
    "ball": "ball",
}


@dataclass(frozen=True)
class Run:
    """One CWRU recording."""

    path: Path
    condition: str  # 'normal' | 'inner_race' | 'outer_race' | 'ball'
    fault_size: float  # inches, 0.0 for healthy
    load: int  # motor load in hp (0-3), a proxy for shaft speed
    clock: int  # outer-race defect position (o'clock); 0 when not applicable
    source_rate: int  # sampling rate as recorded, in Hz
    rate: int  # sampling rate after resampling, in Hz
    signal: np.ndarray  # 1-D drive-end accelerometer trace, at ``rate``

    @property
    def is_healthy(self) -> bool:
        return self.condition == "normal"

    @property
    def label(self) -> str:
        if self.is_healthy:
            return f"normal_{self.load}hp"
        suffix = f"@{self.clock}" if self.clock else ""
        return f"{self.condition}{suffix}_{self.fault_size:.3f}_{self.load}hp"


def _parse_name(path: Path) -> tuple[str, float, int, int] | None:
    """Infer (condition, fault_size, load, clock) from a file name."""
    stem = path.stem
    if stem.isdigit():
        mapped = CWRU_FILE_MAP.get(int(stem))
        # The numeric map covers the 6 o'clock outer-race series only.
        return None if mapped is None else (*mapped, 6 if mapped[0] == "outer_race" else 0)

    match = _DESCRIPTIVE.match(stem.strip())
    if match is None:
        return None
    condition = _CODE_TO_CONDITION.get(match.group("code").lower())
    if condition is None:
        return None
    size = float(match.group("size") or 0) / 1000.0
    load = int(match.group("load") or 0)
    clock = int(match.group("clock") or 0)
    if condition == "normal":
        size = 0.0
    return condition, size, load, clock


def infer_sample_rate(path: Path, condition: str, rules: dict) -> int:
    """Work out the sampling rate of one recording.

    CWRU does not store the sampling rate inside the .mat files, and the archive
    mixes two rates: the drive-end fault recordings are 12 kHz, while the healthy
    baseline recordings are 48 kHz. Training on 48 kHz healthy spectra and
    scoring 12 kHz fault spectra produces an excellent detector of the sampling
    rate and a worthless detector of bearing damage, so the rate is resolved
    explicitly here and logged for every run.
    """
    text = str(path).lower()
    for token, rate in (rules.get("path_rules") or {}).items():
        if token.lower() in text:
            return int(rate)
    if condition == "normal" and rules.get("normal_baseline"):
        return int(rules["normal_baseline"])
    return int(rules.get("default", 12_000))


def _extract_signal(path: Path, channel: str = "DE") -> np.ndarray | None:
    """Pull the accelerometer trace out of a CWRU .mat file.

    Keys look like ``X097_DE_time`` (drive end), ``X097_FE_time`` (fan end).
    """
    mat = loadmat(str(path))
    preferred = [k for k in mat if k.endswith(f"_{channel}_time")]
    fallback = [k for k in mat if k.endswith("_time")]
    keys = preferred or fallback
    if not keys:
        return None
    return np.asarray(mat[sorted(keys)[0]], dtype=np.float64).ravel()


def discover_runs(
    roots: list[str | Path],
    channel: str = "DE",
    min_samples: int = 20_000,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    rate_rules: dict | None = None,
    target_rate: int = 12_000,
) -> list[Run]:
    """Recursively find and load every CWRU .mat file we can label.

    ``include``/``exclude`` are case-insensitive substring filters on the full
    path. Use them to keep one sampling rate: several public mirrors ship the
    48 kHz drive-end recordings alongside the 12 kHz ones, and mixing sampling
    rates silently corrupts every spectral feature in this pipeline.
    """
    runs: list[Run] = []
    seen: set[tuple[str, float, int, int]] = set()

    for mat_path in _iter_mat_files(roots):
        text = str(mat_path).lower()
        if include and not any(token.lower() in text for token in include):
            continue
        if exclude and any(token.lower() in text for token in exclude):
            continue
        parsed = _parse_name(mat_path)
        if parsed is None:
            log.debug("skipping unlabelled file: %s", mat_path)
            continue
        condition, size, load, clock = parsed
        key = (condition, size, load, clock)
        if key in seen:
            continue  # mirrors often duplicate the same run

        signal = _extract_signal(mat_path, channel=channel)
        if signal is None or signal.size < min_samples:
            log.debug("skipping short/empty file: %s", mat_path)
            continue

        source_rate = infer_sample_rate(mat_path, condition, rate_rules or {})
        if source_rate != target_rate:
            signal = resample_poly(signal, target_rate, source_rate)

        seen.add(key)
        runs.append(
            Run(mat_path, condition, size, load, clock, source_rate, target_rate, signal)
        )

    runs.sort(key=lambda r: (r.condition, r.fault_size, r.load, r.clock))
    return runs


def _iter_mat_files(roots: list[str | Path]) -> Iterator[Path]:
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        yield from sorted(root_path.rglob("*.mat"))


def summarise(runs: list[Run]) -> str:
    lines = [
        f"{len(runs)} runs loaded",
        f"{'label':<28} {'samples':>10} {'source Hz':>10} {'seconds':>8}",
    ]
    for run in runs:
        seconds = run.signal.size / run.rate
        lines.append(
            f"{run.label:<28} {run.signal.size:>10,} {run.source_rate:>10,} {seconds:>8.1f}"
        )

    healthy_rates = {r.source_rate for r in runs if r.is_healthy}
    faulty_rates = {r.source_rate for r in runs if not r.is_healthy}
    if healthy_rates and faulty_rates and healthy_rates != faulty_rates:
        lines.append(
            f"NOTE: healthy recorded at {sorted(healthy_rates)} Hz, faults at "
            f"{sorted(faulty_rates)} Hz -- all resampled to {runs[0].rate} Hz."
        )
    lines.append(
        "CHECK: durations should all be roughly 5-25 s. A wildly wrong duration "
        "means the assumed sampling rate for that run is wrong."
    )
    return "\n".join(lines)
