from __future__ import annotations

import json
from typing import Any

import typer

from inspire_aki.config import load_config
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets
from inspire_aki.pipelines.evaluate import run_calibration, run_dca, run_delong, run_metrics
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries
from inspire_aki.pipelines.report import run_consort, run_curves, run_manuscript, run_shap, run_tables
from inspire_aki.pipelines.train import run_train_sequence, run_train_tabular
from inspire_aki.pipelines.tune import run_tune_sequence, run_tune_tabular
from inspire_aki.runtime import runtime_summary


app = typer.Typer(help="CLI entrypoint for the refactored INSPIRE AKI pipeline.")
preprocess_app = typer.Typer(help="Preprocessing stages.")
tune_app = typer.Typer(help="Hyperparameter optimization stages.")
train_app = typer.Typer(help="Model training stages.")
evaluate_app = typer.Typer(help="Evaluation stages.")
report_app = typer.Typer(help="Report generation stages.")
explain_app = typer.Typer(help="Interpretability stages.")
compat_app = typer.Typer(help="Compatibility exports.")
run_app = typer.Typer(help="Orchestrated multi-stage runs.")
runtime_app = typer.Typer(help="Runtime inspection and resource planning.")

app.add_typer(preprocess_app, name="preprocess")
app.add_typer(tune_app, name="tune")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(report_app, name="report")
app.add_typer(explain_app, name="explain")
app.add_typer(compat_app, name="compat")
app.add_typer(run_app, name="run")
app.add_typer(runtime_app, name="runtime")


def _cfg(config_path: str | None) -> dict[str, Any]:
    return load_config(config_path)


def _echo(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@preprocess_app.command("preop")
def preprocess_preop(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_preop(_cfg(config)))


@preprocess_app.command("intraop")
def preprocess_intraop(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_intraop(_cfg(config)))


@preprocess_app.command("tabular")
def preprocess_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_tabular(_cfg(config)))


@preprocess_app.command("labels")
def preprocess_labels(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_labels(_cfg(config)))


@preprocess_app.command("timeseries")
def preprocess_timeseries(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_timeseries(_cfg(config)))


@preprocess_app.command("sequence")
def preprocess_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_sequence(_cfg(config)))


@tune_app.command("tabular")
def tune_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_tune_tabular(_cfg(config)))


@tune_app.command("sequence")
def tune_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_tune_sequence(_cfg(config)))


@train_app.command("tabular")
def train_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_train_tabular(_cfg(config)))


@train_app.command("sequence")
def train_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_train_sequence(_cfg(config)))


@evaluate_app.command("calibrate")
def evaluate_calibrate(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_calibration(_cfg(config)))


@evaluate_app.command("metrics")
def evaluate_metrics(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_metrics(_cfg(config)))


@evaluate_app.command("delong")
def evaluate_delong(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_delong(_cfg(config)))


@evaluate_app.command("dca")
def evaluate_dca(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_dca(_cfg(config)))


@explain_app.command("shap")
def explain_shap(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_shap(_cfg(config)))


@report_app.command("consort")
def report_consort(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_consort(_cfg(config)))


@report_app.command("tables")
def report_tables(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_tables(_cfg(config)))


@report_app.command("curves")
def report_curves(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_curves(_cfg(config)))


@report_app.command("manuscript")
def report_manuscript(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(run_manuscript(_cfg(config)))


@compat_app.command("export-legacy")
def compat_export_legacy(config: str | None = typer.Option(None, "--config")) -> None:
    cfg = _cfg(config)
    artifacts = ArtifactManager(cfg)
    _echo({"outputs": [str(path) for path in export_legacy_datasets(artifacts)]})


@runtime_app.command("inspect")
def runtime_inspect(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(runtime_summary(_cfg(config)))


@run_app.command("all")
def run_all(config: str | None = typer.Option(None, "--config")) -> None:
    cfg = _cfg(config)
    outputs: dict[str, Any] = {
        "preprocess_preop": run_preop(cfg),
        "preprocess_intraop": run_intraop(cfg),
        "preprocess_tabular": run_tabular(cfg),
        "preprocess_labels": run_labels(cfg),
        "preprocess_timeseries": run_timeseries(cfg),
        "preprocess_sequence": run_sequence(cfg),
        "tune_tabular": run_tune_tabular(cfg),
        "tune_sequence": run_tune_sequence(cfg),
        "train_tabular": run_train_tabular(cfg),
        "train_sequence": run_train_sequence(cfg),
        "evaluate_calibrate": run_calibration(cfg),
        "evaluate_metrics": run_metrics(cfg),
        "evaluate_delong": run_delong(cfg),
        "evaluate_dca": run_dca(cfg),
        "report_manuscript": run_manuscript(cfg),
    }
    _echo(outputs)


def main() -> None:
    app()
