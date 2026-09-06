"""Turn fused scores into probabilities and a decision tier.

Confidence = softmax over the top candidates' fused scores with a temperature
fitted on a validation set (``mcv evaluate --fit-calibration``). The tier rules
give the caller an actionable answer instead of a raw number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mcmaster_vision.schemas import MatchTier


@dataclass
class Calibration:
    temperature: float = 0.05  # softmax temperature over fused scores
    exact_threshold: float = 0.90  # p(best) above this and margin -> EXACT
    likely_threshold: float = 0.60
    unknown_similarity: float = 0.35  # best cosine below this -> UNKNOWN
    min_margin: float = 0.15  # p(best) - p(second) needed for EXACT

    def probabilities(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        z = np.asarray(scores, np.float64) / max(self.temperature, 1e-6)
        z -= z.max()
        e = np.exp(z)
        return (e / e.sum()).tolist()

    def tier(self, probs: list[float], best_similarity: float, ocr_hit: bool = False) -> MatchTier:
        if ocr_hit:
            return MatchTier.EXACT
        if not probs or best_similarity < self.unknown_similarity:
            return MatchTier.UNKNOWN
        p0 = probs[0]
        margin = p0 - (probs[1] if len(probs) > 1 else 0.0)
        if p0 >= self.exact_threshold and margin >= self.min_margin:
            return MatchTier.EXACT
        if p0 >= self.likely_threshold:
            return MatchTier.LIKELY
        return MatchTier.CANDIDATE

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def fit_thresholds(
        self,
        score_lists: list[list[float]],
        correct_idx: list[int],
        *,
        exact_precision: float = 0.98,
        likely_precision: float = 0.90,
        min_support: int = 10,
    ) -> Calibration:
        """Choose ``exact_threshold`` / ``likely_threshold`` (on the calibrated top-1
        probability) so that answers in each tier meet a *precision target* on the
        validation queries. Returns a new Calibration; thresholds stay at their
        defaults when no threshold reaches the target with ``min_support`` answers."""
        rows = []
        for scores, ci in zip(score_lists, correct_idx, strict=True):
            if not scores:
                continue
            probs = self.probabilities(scores)
            margin = probs[0] - (probs[1] if len(probs) > 1 else 0.0)
            rows.append((probs[0], margin, ci == 0))
        if not rows:
            return Calibration(**self.__dict__)

        def best_threshold(target: float, need_margin: bool) -> float | None:
            cands = sorted({round(p, 3) for p, _, _ in rows})
            chosen = None
            for thr in cands:  # lowest threshold that still meets the target = most coverage
                sel = [
                    ok
                    for p, m, ok in rows
                    if p >= thr and (m >= self.min_margin if need_margin else True)
                ]
                if len(sel) >= min_support and sum(sel) / len(sel) >= target:
                    chosen = thr
                    break
            return chosen

        exact = best_threshold(exact_precision, need_margin=True)
        likely = best_threshold(likely_precision, need_margin=False)
        new = Calibration(**self.__dict__)
        if exact is not None:
            new.exact_threshold = float(exact)
        if likely is not None:
            new.likely_threshold = float(min(likely, new.exact_threshold))
        return new

    @classmethod
    def fit_temperature(
        cls, score_lists: list[list[float]], correct_idx: list[int], grid=None
    ) -> Calibration:
        """Pick the temperature minimising NLL of the correct candidate on validation queries."""
        grid = grid or [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
        best_t, best_nll = 0.05, float("inf")
        for t in grid:
            cal = cls(temperature=t)
            nll = 0.0
            n = 0
            for scores, ci in zip(score_lists, correct_idx, strict=True):
                if ci < 0 or not scores:
                    continue
                p = cal.probabilities(scores)[ci]
                nll -= np.log(max(p, 1e-9))
                n += 1
            if n and nll / n < best_nll:
                best_t, best_nll = t, nll / n
        return cls(temperature=best_t)
