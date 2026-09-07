"""Layouts from the stainless fittings pages (bushings A x B, max psi per column, butt-weld
wall thickness, flanges, thread adapters, pipe by the foot) go through the importer."""

from __future__ import annotations

from mcmaster_vision.catalog.pages import parse_page

BUSHINGS = """Stainless Steel Pipe Fittings & Flanges
High-Pressure Stainless Steel Threaded Pipe Fittings
Hex Reducing Bushings, Male x Female
Pipe Size (A)     (B)
Type 304 Stainless Steel
1/4 ........ 1/8 ........ 4464K381 .... $4.40
3/8 ........ 1/8 ........ 4464K641 ..... 4.73
1/2 ........ 1/4 ........ 4464K397 ..... 6.30
Type 316 Stainless Steel
1/4 ........ 1/8 ........ 4443K731 .... $4.79
13
McMASTER-CARR
"""

ELBOWS_PSI = """Stainless Steel Pipe Fittings
Extreme-Pressure Stainless Steel Threaded Pipe Fittings
Connections: NPTF (Dryseal), unless noted. NPTF threads are compatible with NPT threads.
Elbows, Tees, and Crosses
Pipe Size     Max. psi @ 72° F     90° Elbows, Female     Max. psi @ 72° F     90° Elbows, Male     Max. psi @ 72° F     45° Elbows, Female x Male
1/8 ........ 5,000 ...... 51205K162 .... $21.99     6,000 ...... 51205K113 .... $33.73     5,000 ...... 51205K119 .... $29.19
1/4 ........ 5,000 ...... 51205K152 ..... 24.63     6,000 ...... 51205K112 ..... 18.67     5,000 ...... 51205K121 ..... 35.25
16
McMASTER-CARR
"""

BUTT_WELD = """Stainless Steel Pipe Fittings
Thin-Wall Butt-Weld Stainless Steel Unthreaded Pipe Fittings
Connections: Butt weld (unthreaded).
Pipe Size     Wall Thick.     (C)     90° Elbows, Long Radius     45° Elbows     Tees
Type 304/304L Stainless Steel
1/2 ........ 0.083" ...... 1-1/2" ...... 45735K211 .... $5.33     45735K231 .... $6.83     45735K251 .... $18.83
2 .......... 0.109" ...... 3" .......... 45735K216 ..... 9.17     45735K236 ..... 9.36     45735K256 ..... 20.89
20
McMASTER-CARR
"""

FLANGES = """Stainless Steel Pipe Fittings & Flanges
High-Pressure Stainless Steel Threaded Flanges
Connections: NPT.
Pipe Size     Flange OD     Qty.     Dia.     Type 304/304L Stainless Steel
1/2 ........ 3-3/4" ...... 4 ...... 1/2" ...... 7977K11 .... $41.19
3/4 ........ 4-5/8" ...... 4 ...... 5/8" ...... 7977K12 ..... 42.84
13
McMASTER-CARR
"""

ADAPTERS = """Stainless Steel Pipe Fittings
Extreme-Pressure Stainless Steel Threaded Adapters
Connections: BSPP (British Standard Pipe Parallel), NPT, metric, NPTF (Dryseal), or UN/UNF straight threads.
Female x Male Adapters
NPT (A) x Metric (B)
Pipe Size (A)     Thread Size (B)     Max. psi @ 72° F
1/8 ........ M10 x 1.0 ...... 6,000 ...... 4822T61 .... $40.32
1/4 ........ M12 x 1.5 ...... 6,000 ...... 4822T62 ..... 29.04
17
McMASTER-CARR
"""

PIPE_BY_FOOT = """Stainless Steel Pipe Fittings & Pipe
Thin-Wall Stainless Steel Unthreaded Pipe
Pipe Size     1 ft.     3 ft.     6 ft.
Type 304/304L Stainless Steel
1/2 ........ 4347K31 .... $10.70     4347K32 .... $21.74     4347K33 .... $33.44
3/4 ........ 4347K34 ..... 12.77     4347K35 ..... 25.93     4347K36 ..... 39.90
21
McMASTER-CARR
"""


def _by_pn(text):
    return {p.part_number: p for p in parse_page(text, "p")}


def test_reducing_bushings_carry_both_sizes():
    parts = _by_pn(BUSHINGS)
    b = parts["4464K381"]
    assert b.attributes["pipe_size"] == "1/4" and b.attributes["pipe_size_b"] == "1/8"
    assert "length" not in b.attributes and b.attributes["material"] == "Type 304 Stainless Steel"
    assert "1/4 x 1/8" in b.name and "Bushing" in b.name
    assert parts["4443K731"].attributes["material"] == "Type 316 Stainless Steel"
    assert len(parts) == 4


def test_max_psi_per_column_and_connection():
    parts = _by_pn(ELBOWS_PSI)
    assert parts["51205K162"].attributes["max_psi"] == "5000"
    assert parts["51205K113"].attributes["max_psi"] == "6000"
    assert parts["51205K113"].attributes["fitting_type"].startswith("90° Elbows, Male")
    assert parts["51205K119"].attributes["fitting_type"].startswith("45° Elbows")
    assert parts["51205K162"].attributes["connection"].startswith("NPTF")
    assert "length" not in parts["51205K162"].attributes


def test_butt_weld_wall_thickness_and_dimension():
    parts = _by_pn(BUTT_WELD)
    e = parts["45735K211"]
    assert e.attributes["wall_thickness"] == '0.083"' and e.attributes["dimension_c"] == '1-1/2"'
    assert e.attributes["connection"].lower().startswith("butt weld")
    assert parts["45735K256"].attributes["fitting_type"] == "Tees"
    assert parts["45735K216"].attributes["wall_thickness"] == '0.109"'


def test_flanges_bolt_columns():
    parts = _by_pn(FLANGES)
    f = parts["7977K11"]
    assert f.attributes["flange_od"] == '3-3/4"' and f.attributes["bolt_qty"] == "4"
    assert f.attributes["bolt_dia"] == '1/2"'


def test_thread_adapters_name_both_standards():
    parts = _by_pn(ADAPTERS)
    a = parts["4822T61"]
    assert a.attributes["thread_b"] == "M10 x 1.0" and a.attributes["max_psi"] == "6000"
    assert a.attributes["thread_a"] == "NPT" and a.attributes["thread_b_type"] == "METRIC"


def test_pipe_sold_by_the_foot():
    parts = _by_pn(PIPE_BY_FOOT)
    assert parts["4347K31"].attributes["length"] == "1 ft"
    assert parts["4347K36"].attributes["length"] == "6 ft"
    assert parts["4347K31"].attributes["pipe_size"] == "1/2"


OUTLETS = """Stainless Steel Pipe Fittings
High-Pressure Stainless Steel Threaded Pipe Outlets
Connections: NPT.
Fits Pipe Size Range     Outlet Pipe Size     Ht.     Max. psi @ 72° F     Type 304/304L Stainless Steel     Type 316/316L Stainless Steel
1/4 to 36 ........ 1/4 ...... 3/4" ...... 3,000 ...... 4565T31 .... $28.50     4583T11 .... $33.24
1/2 to 36 ........ 1/2 ...... 1" ........ 3,000 ...... 4565T33 ..... 29.80     4583T13 ..... 34.65
16
McMASTER-CARR
"""


def test_pipe_outlets_fit_a_size_range():
    parts = _by_pn(OUTLETS)
    o = parts["4565T31"]
    assert o.attributes["pipe_size"] == "1/4 to 36" and o.attributes["outlet_pipe_size"] == "1/4"
    assert o.attributes["height"] == '3/4"' and o.attributes["max_psi"] == "3000"
    assert o.attributes["fitting_type"].startswith("Type 304")
    assert parts["4583T13"].attributes["fitting_type"].startswith("Type 316")
