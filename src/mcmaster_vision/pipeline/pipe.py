"""Pipe sizing knowledge from the catalog's "Selecting and Measuring Pipe & Fittings" pages.

A *pipe size* ("3/8", "1-1/2") is a nominal industry designation, not a measurement: a
3/8 fitting is 0.675" across the male threads. So a measured dimension must be compared
with the nominal size's outside diameter (male threads / pipe OD) or inside diameter
(female threads), never with the nominal number itself. Thread pitch (threads per inch)
separates NPT from BSP at the same size, and the compatibility table says which thread
types mate.
"""

from __future__ import annotations

import re
from fractions import Fraction

INCH = 25.4

# nominal pipe size -> outside diameter of the pipe / male threads (inches, ANSI B36.10)
PIPE_OD_IN: dict[str, float] = {
    "1/16": 0.3125,
    "1/8": 0.405,
    "1/4": 0.540,
    "3/8": 0.675,
    "1/2": 0.840,
    "3/4": 1.050,
    "1": 1.315,
    "1-1/4": 1.660,
    "1-1/2": 1.900,
    "2": 2.375,
    "2-1/2": 2.875,
    "3": 3.500,
    "3-1/2": 4.000,
    "4": 4.500,
    "5": 5.563,
    "6": 6.625,
    "8": 8.625,
}

# schedule 40 (standard wall) inside diameter, inches: what a female fitting measures
PIPE_ID_SCH40_IN: dict[str, float] = {
    "1/8": 0.269,
    "1/4": 0.364,
    "3/8": 0.493,
    "1/2": 0.622,
    "3/4": 0.824,
    "1": 1.049,
    "1-1/4": 1.380,
    "1-1/2": 1.610,
    "2": 2.067,
    "2-1/2": 2.469,
    "3": 3.068,
    "4": 4.026,
    "5": 5.047,
    "6": 6.065,
    "8": 7.981,
}

# wall thickness by schedule, inches: schedule 10 ("thin-wall") and 40 ("standard-wall")
# from the catalog's butt-weld fitting tables; schedule 80 ("thick-wall", high-pressure
# nipples) from ANSI B36.10. A female or unthreaded fitting's inside diameter is the
# pipe OD minus two walls of its schedule.
PIPE_WALL_IN: dict[str, dict[str, float]] = {
    "10": {
        "1/8": 0.049,
        "1/4": 0.065,
        "3/8": 0.065,
        "1/2": 0.083,
        "3/4": 0.083,
        "1": 0.109,
        "1-1/4": 0.109,
        "1-1/2": 0.109,
        "2": 0.109,
        "2-1/2": 0.120,
        "3": 0.120,
        "4": 0.120,
        "6": 0.134,
        "8": 0.148,
    },
    "40": {
        "1/8": 0.068,
        "1/4": 0.088,
        "3/8": 0.091,
        "1/2": 0.109,
        "3/4": 0.113,
        "1": 0.133,
        "1-1/4": 0.140,
        "1-1/2": 0.145,
        "2": 0.154,
        "2-1/2": 0.203,
        "3": 0.216,
        "4": 0.237,
        "6": 0.280,
        "8": 0.322,
    },
    "80": {
        "1/8": 0.095,
        "1/4": 0.119,
        "3/8": 0.126,
        "1/2": 0.147,
        "3/4": 0.154,
        "1": 0.179,
        "1-1/4": 0.191,
        "1-1/2": 0.200,
        "2": 0.218,
        "2-1/2": 0.276,
        "3": 0.300,
        "4": 0.337,
        "6": 0.432,
        "8": 0.500,
    },
}

# threads per inch by pipe size: (NPT, BSP)
THREADS_PER_INCH: dict[str, tuple[float | None, float | None]] = {
    "1/16": (27, None),
    "1/8": (27, 28),
    "1/4": (18, 19),
    "3/8": (18, 19),
    "1/2": (14, 14),
    "5/8": (None, 14),
    "3/4": (14, 14),
    "1": (11.5, 11),
    "1-1/4": (11.5, 11),
    "1-1/2": (11.5, 11),
    "2": (11.5, 11),
    "2-1/2": (8, 11),
    "3": (8, 11),
    "3-1/2": (8, 11),
    "4": (8, 11),
    "5": (8, 11),
    "6": (8, 11),
    "8": (8, None),
}

# thread type -> the thread types a MALE of it mates with (female side), per the catalog's
# compatibility table; straight threads need a seal, tapered ones seal on the thread
THREAD_COMPATIBILITY: dict[str, dict[str, list[str]]] = {
    "NPT": {
        "male_fits": ["NPT", "NPTF", "NPSM", "NPSH", "NPSL", "NPSC"],
        "female_fits": ["NPT", "NPTF"],
    },
    "NPTF": {"male_fits": ["NPTF", "NPT", "NPSM", "NPSH"], "female_fits": ["NPTF", "NPT"]},
    "BSPT": {"male_fits": ["BSPT", "BSPP"], "female_fits": ["BSPT"]},
    "BSPP": {"male_fits": ["BSPP"], "female_fits": ["BSPP", "BSPT"]},
    "NPSM": {"male_fits": ["NPSM", "NPSH"], "female_fits": ["NPSM", "NPT", "NPTF"]},
    "NPSH": {"male_fits": ["NPSH"], "female_fits": ["NPSH", "NPT", "NPTF", "NPSM"]},
    "NPSL": {"male_fits": [], "female_fits": ["NPT"]},
    "NPSC": {"male_fits": [], "female_fits": ["NPT"]},
    "NH/NST": {"male_fits": ["NH/NST"], "female_fits": ["NH/NST"]},
    "GHT": {"male_fits": ["GHT"], "female_fits": ["GHT"]},
    "UN/UNF": {"male_fits": ["UN/UNF"], "female_fits": ["UN/UNF"]},
    "Metric DIN 3852": {"male_fits": ["Metric DIN 3852"], "female_fits": ["Metric DIN 3852"]},
    "Metric DIN 3901/3902": {
        "male_fits": ["Metric DIN 3901/3902"],
        "female_fits": ["Metric DIN 3901/3902"],
    },
}
TAPERED = {"NPT", "NPTF", "BSPT"}

_SIZE = re.compile(
    r"^(\d+)?\s*[- ]?\s*(\d+/\d+)?\s*(?:\"|″|”|in\b|inch|inches|)?\s*(?:npt|bspt|bspp|nptf)?$", re.I
)


def normalise_pipe_size(text) -> str | None:
    """``'3/8"'``, ``"1 1/4 NPT"``, ``"1-1/2"`` -> the catalog key (``"3/8"``, ``"1-1/4"``)."""
    t = str(text).strip().lower().replace("″", '"').replace("”", '"')
    t = re.sub(r"\s*\b(pipe|size|thread|nptf|npt|bspt|bspp)\b\s*", " ", t).strip(" .")
    m = _SIZE.match(t)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    whole, frac = m.group(1), m.group(2)
    if frac:
        num, den = frac.split("/")
        if int(den) == 0:
            return None
        f = Fraction(int(num), int(den))
        if f >= 1 or f.denominator not in (2, 4, 8, 16):
            return None
        frac = f"{f.numerator}/{f.denominator}"
    key = f"{whole}-{frac}" if whole and frac else (whole or frac)
    return key if key in PIPE_OD_IN or key in THREADS_PER_INCH else None


def pipe_od_mm(size) -> float | None:
    key = normalise_pipe_size(size)
    return round(PIPE_OD_IN[key] * INCH, 2) if key in PIPE_OD_IN else None


def pipe_size_from_od_mm(od_mm: float, tolerance: float = 0.12) -> str | None:
    """The nominal size whose OD is nearest (within ``tolerance`` relative), else None."""
    best, err = None, tolerance
    for key, od in PIPE_OD_IN.items():
        e = abs(od_mm / (od * INCH) - 1.0)
        if e < err:
            best, err = key, e
    return best


def pipe_size_from_id_mm(id_mm: float, tolerance: float = 0.12) -> str | None:
    best, err = None, tolerance
    for key, i in PIPE_ID_SCH40_IN.items():
        e = abs(id_mm / (i * INCH) - 1.0)
        if e < err:
            best, err = key, e
    return best


def thread_family_from_pitch(size, threads_per_inch: float) -> list[str]:
    """Which of NPT / BSP has this pitch at this size (both when they share it)."""
    key = normalise_pipe_size(size)
    if key not in THREADS_PER_INCH:
        return []
    npt, bsp = THREADS_PER_INCH[key]
    out = []
    if npt and abs(threads_per_inch - npt) / npt < 0.025:  # 27 vs 28 tpi are 3.6% apart
        out.append("NPT")
    if bsp and abs(threads_per_inch - bsp) / bsp < 0.025:
        out.append("BSP")
    return out


def compatible_threads(thread_type: str, gender: str = "male") -> list[str]:
    """What a ``gender`` thread of ``thread_type`` mates with (per the catalog table)."""
    t = thread_type.strip().upper()
    for name, row in THREAD_COMPATIBILITY.items():
        if name.upper() == t:
            return list(row["male_fits" if gender.lower().startswith("m") else "female_fits"])
    return []


def is_tapered(thread_type: str) -> bool | None:
    t = thread_type.strip().upper()
    if t in TAPERED:
        return True
    if t in {k.upper() for k in THREAD_COMPATIBILITY}:
        return False
    return None


def schedule_from_text(text: str) -> str | None:
    """``"Schedule 40"``, ``"Sch. 80"``, ``"thin-wall"``, ``"thick-wall"``, ``"standard-wall"``
    -> the schedule key, else None."""
    t = str(text).lower()
    m = re.search(r"sch(?:edule)?\.?\s*(10|40|80)\b", t)
    if m:
        return m.group(1)
    if "thin-wall" in t or "thin wall" in t:
        return "10"
    if "thick-wall" in t or "thick wall" in t or "extra-heavy" in t:
        return "80"
    if "standard-wall" in t or "standard wall" in t:
        return "40"
    return None


def pipe_id_mm(size, schedule: str | None = None, wall_in: float | None = None) -> float | None:
    """Inside diameter of a nominal size: from an explicit wall thickness (inches), else
    the schedule's wall (default 40), else the schedule-40 table."""
    key = normalise_pipe_size(size)
    if key is None:
        return None
    od = PIPE_OD_IN.get(key)
    if od is None:
        return None
    if wall_in is None:
        wall_in = PIPE_WALL_IN.get(schedule or "40", {}).get(key)
    if wall_in is not None:
        return round((od - 2 * wall_in) * INCH, 2)
    inner = PIPE_ID_SCH40_IN.get(key)
    return round(inner * INCH, 2) if inner else None
