from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from inspire_aki.cli import app


def test_run_all_smoke(synthetic_config: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])
    assert result.exit_code == 0, result.stdout

    artifacts_dir = synthetic_config.parent / "artifacts"
    assert (artifacts_dir / "predictions" / "calibrated_predictions.parquet").exists()
    assert (artifacts_dir / "evaluation" / "metrics_summary.csv").exists()
    assert (artifacts_dir / "reports" / "tables" / "performance_table.csv").exists()
    assert (artifacts_dir / "reports" / "figures" / "roc_curves_preop.png").exists()

