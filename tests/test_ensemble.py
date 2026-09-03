from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mcmaster_vision.models import HashBackbone
from mcmaster_vision.models.ensemble import EnsembleBackbone


def test_ensemble_cosine_is_weighted_mean_of_member_cosines():
    a, b = HashBackbone(thumb=8), HashBackbone(thumb=12)
    ens = EnsembleBackbone([a, b], weights=[3.0, 1.0])
    assert ens.dim == a.dim + b.dim
    imgs = [
        Image.effect_noise((64, 64), 60).convert("RGB"),
        Image.effect_noise((64, 64), 90).convert("RGB"),
    ]
    e = ens.embed(imgs)
    assert e.shape == (2, ens.dim) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    ea, eb = a.embed(imgs), b.embed(imgs)
    expected = 0.75 * float(ea[0] @ ea[1]) + 0.25 * float(eb[0] @ eb[1])
    assert abs(float(e[0] @ e[1]) - expected) < 1e-4
    assert "ensemble(" in ens.version
    with pytest.raises(ValueError):
        EnsembleBackbone([a], weights=[1, 2])
