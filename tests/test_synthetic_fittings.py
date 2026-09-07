"""The pipe-fitting kinds added from the catalog's fittings pages."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mcmaster_vision.data.synthetic import FAMILY_KINDS, SyntheticCatalog
from mcmaster_vision.pipeline.pipe import PIPE_OD_IN, compatible_threads, normalise_pipe_size

FITTINGS = ["pipe_nipple", "pipe_coupling", "pipe_flange", "pipe_cap", "pipe_bushing"]


def test_fittings_render_distinct_views_with_catalog_attributes(tmp_path):
    assert set(FITTINGS) <= set(FAMILY_KINDS)
    parts = list(
        SyntheticCatalog(n_parts=15, images_per_part=3, seed=11, kinds=FITTINGS).generate(tmp_path)
    )
    kinds_seen = {p.family_id.split(":")[0] for p in parts}
    assert kinds_seen == set(FITTINGS)
    for p in parts:
        a = p.attributes
        assert a["pipe_size"] and a["thread_type"] in ("NPT", "NPTF", "BSPT")
        assert normalise_pipe_size(a["pipe_size"]) is not None
        assert a["gender"] == ("male" if "Nipple" in p.name else "female")
        assert compatible_threads(a["thread_type"], a["gender"])
        if "Nipple" in p.name:
            assert a["length"]
        if "Bushing" in p.name:
            small, big = normalise_pipe_size(a["reduced_to"]), normalise_pipe_size(a["pipe_size"])
            assert PIPE_OD_IN[small] < PIPE_OD_IN[big]
        views = [np.asarray(Image.open(x).convert("L"), dtype=np.float32) for x in p.image_paths]
        assert len(views) == 3 and all(v.min() < 200 for v in views)  # something is drawn
        # the three views differ (a top view or a rotation), never three copies
        assert np.abs(views[0] - views[1]).mean() > 1 and np.abs(views[0] - views[2]).mean() > 1
