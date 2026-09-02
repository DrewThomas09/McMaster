from __future__ import annotations

from typer.testing import CliRunner

from mcmaster_vision.cli import app


def test_demo_end_to_end(tmp_path):
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "demo",
            "--parts",
            "25",
            "--images-per-part",
            "2",
            "--data-dir",
            str(tmp_path),
            "--no-serve",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "recall_at" in r.output
    assert (tmp_path / "index" / "parts" / "meta.json").exists()
    assert (tmp_path / "sample_query.jpg").exists()

    q = tmp_path / "sample_query.jpg"
    r = runner.invoke(
        app,
        ["identify", str(q), "--top-n", "3"],
        env={
            "MCV_CATALOG_DB": str(tmp_path / "catalog.sqlite"),
            "MCV_INDEX_DIR": str(tmp_path / "index"),
            "MCV_MODEL_DIR": str(tmp_path / "models"),
        },
    )
    assert r.exit_code == 0, r.output
    assert '"candidates"' in r.output
