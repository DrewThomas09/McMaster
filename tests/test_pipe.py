from __future__ import annotations

import pytest

from mcmaster_vision.pipeline.measure import Measurement, size_consistency
from mcmaster_vision.pipeline.pipe import (
    compatible_threads,
    is_tapered,
    normalise_pipe_size,
    pipe_id_mm,
    pipe_od_mm,
    pipe_size_from_id_mm,
    pipe_size_from_od_mm,
    thread_family_from_pitch,
)
from mcmaster_vision.schemas import Part


@pytest.mark.parametrize(
    "text,key",
    [
        ('3/8"', "3/8"),
        ("3/8", "3/8"),
        ("1 1/4 NPT", "1-1/4"),
        ("1-1/2", "1-1/2"),
        ('2"', "2"),
        ("1/16", "1/16"),
        ("M6", None),
        ("", None),
        ("7/9", None),
    ],
)
def test_normalise_pipe_size(text, key):
    assert normalise_pipe_size(text) == key


def test_nominal_size_maps_to_real_diameters():
    assert pipe_od_mm('3/8"') == pytest.approx(17.15, abs=0.05)  # 0.675" per the catalog scale
    assert pipe_id_mm("3/8") == pytest.approx(12.52, abs=0.05)
    assert pipe_size_from_od_mm(17.0) == "3/8" and pipe_size_from_od_mm(33.4) == "1"
    assert pipe_size_from_id_mm(15.8) == "1/2"
    assert pipe_size_from_od_mm(500) is None


def test_thread_pitch_and_compatibility():
    assert thread_family_from_pitch("1/8", 27) == ["NPT"]
    assert thread_family_from_pitch("1/8", 28) == ["BSP"]
    assert thread_family_from_pitch("1/2", 14) == ["NPT", "BSP"]
    assert thread_family_from_pitch("3", 8) == ["NPT"] and thread_family_from_pitch("3", 11) == [
        "BSP"
    ]
    assert "NPSM" in compatible_threads("NPT", "male") and compatible_threads("NPT", "female") == [
        "NPT",
        "NPTF",
    ]
    assert compatible_threads("BSPP", "male") == ["BSPP"]
    assert is_tapered("NPT") is True and is_tapered("BSPP") is False and is_tapered("weird") is None


def test_size_matching_uses_pipe_od_not_the_nominal_number():
    # a 3/8 male elbow photographed next to a coin: 19 mm across the body
    m = Measurement(long_mm=30.0, short_mm=19.0, mm_per_px=0.1)
    elbow_38 = Part(
        part_number="A",
        name="90° Elbow, NPT male",
        category_path=["x"],
        attributes={"pipe_size": '3/8"'},
    )
    elbow_34 = Part(
        part_number="B",
        name="90° Elbow, NPT male",
        category_path=["x"],
        attributes={"pipe_size": '3/4"'},
    )
    elbow_18 = Part(
        part_number="C",
        name="90° Elbow, NPT male",
        category_path=["x"],
        attributes={"pipe_size": '1/8"'},
    )
    assert size_consistency(m, elbow_38)[0] == 1.0
    assert size_consistency(m, elbow_34)[0] < 0  # 3/4 is 26.7 mm OD: too big
    assert size_consistency(m, elbow_18)[0] < 0  # 1/8 is 10.3 mm OD, body at most ~19.5
    assert "pipe OD" in size_consistency(m, elbow_38)[1][0]
    # a female coupling: the bore is what shows
    coupling = Part(
        part_number="D",
        name="Coupling, female NPT",
        category_path=["x"],
        attributes={"pipe_size": '1/2"'},
    )
    assert size_consistency(Measurement(25.0, 16.0, 0.1), coupling)[0] == 1.0
