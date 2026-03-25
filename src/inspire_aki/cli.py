from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import typer

from inspire_aki.benchmarking import run_runtime_benchmarks
from inspire_aki.config import REPO_ROOT, load_config
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets
from inspire_aki.io.progress import ProgressLogger
from inspire_aki.orchestration import OverlapInterruptedError, run_overlap_stages
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


def _parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_cli_path(config_path: str | None) -> str | None:
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _progress_logger(config: dict[str, Any], *, stdout: bool = True) -> ProgressLogger | None:
    if "paths" not in config:
        return None
    return ProgressLogger(ArtifactManager(config), ("logs", "run_all_events.jsonl"), stdout=stdout)


def _aborted_stdout_message(stage_name: str) -> str:
    return f"Interrupted {stage_name}; exiting cleanly (130)."


def _run_command(*, stage_name: str, config_path: str | None, runner: Any) -> None:
    try:
        _echo(runner(_cfg(config_path)))
    except KeyboardInterrupt as exc:
        typer.echo(_aborted_stdout_message(stage_name), err=True)
        raise typer.Exit(code=130) from exc


def _run_stage(
    *,
    stage_name: str,
    runner: Any,
    config: dict[str, Any],
    progress: ProgressLogger | None,
) -> dict[str, Any]:
    start = perf_counter()
    if progress is not None:
        progress.stage_start(stage_name)
    try:
        payload = runner(config)
    except KeyboardInterrupt as exc:
        if progress is not None:
            progress.stage_error(
                stage_name,
                error="KeyboardInterrupt",
                status="aborted",
                wall_time_seconds=perf_counter() - start,
            )
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        if progress is not None:
            progress.stage_error(stage_name, error=f"{type(exc).__name__}: {exc}", wall_time_seconds=perf_counter() - start)
        raise
    if progress is not None:
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start)
    return payload


def _run_overlap_branch(
    *,
    config: dict[str, Any],
    config_path: str | None,
    progress: ProgressLogger | None,
) -> dict[str, dict[str, Any]]:
    if progress is None:
        raise RuntimeError("Overlap orchestration requires artifact-backed configuration.")
    log_dir = progress.artifacts.resolve("logs")
    config_cli_path = _config_cli_path(config_path)
    stages = ["tune_sequence", "train_tabular"]
    for stage_name in stages:
        progress.stage_start(stage_name, message="launched as overlapped subprocess")
    try:
        results = run_overlap_stages(stages, config_path=config_cli_path, log_dir=log_dir)
        interrupted = False
    except OverlapInterruptedError as exc:
        results = exc.results
        interrupted = True

    outputs: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, int, Path]] = []
    for stage_name in stages:
        result = results[stage_name]
        outputs[stage_name] = result.payload
        if result.returncode == 0:
            progress.stage_end(
                stage_name,
                wall_time_seconds=result.wall_time_seconds,
                message=f"subprocess completed; log={result.log_path}",
            )
            continue
        failures.append((stage_name, result.returncode, result.log_path))
        progress.stage_error(
            stage_name,
            error=f"subprocess exited with code {result.returncode}; log={result.log_path}",
            status="aborted" if interrupted else "error",
            wall_time_seconds=result.wall_time_seconds,
        )
    if interrupted:
        raise typer.Exit(code=130)
    if failures:
        first_stage, returncode, _ = failures[0]
        raise typer.Exit(code=returncode or 1) from RuntimeError(f"Overlapped stage failed: {first_stage}")
    return outputs


@preprocess_app.command("preop")
def preprocess_preop(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_preop", config_path=config, runner=run_preop)


@preprocess_app.command("intraop")
def preprocess_intraop(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_intraop", config_path=config, runner=run_intraop)


@preprocess_app.command("tabular")
def preprocess_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_tabular", config_path=config, runner=run_tabular)


@preprocess_app.command("labels")
def preprocess_labels(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_labels", config_path=config, runner=run_labels)


@preprocess_app.command("timeseries")
def preprocess_timeseries(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_timeseries", config_path=config, runner=run_timeseries)


@preprocess_app.command("sequence")
def preprocess_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="preprocess_sequence", config_path=config, runner=run_sequence)


@tune_app.command("tabular")
def tune_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="tune_tabular", config_path=config, runner=run_tune_tabular)


@tune_app.command("sequence")
def tune_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="tune_sequence", config_path=config, runner=run_tune_sequence)


@train_app.command("tabular")
def train_tabular(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="train_tabular", config_path=config, runner=run_train_tabular)


@train_app.command("sequence")
def train_sequence(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="train_sequence", config_path=config, runner=run_train_sequence)


@evaluate_app.command("calibrate")
def evaluate_calibrate(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="evaluate_calibrate", config_path=config, runner=run_calibration)


@evaluate_app.command("metrics")
def evaluate_metrics(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="evaluate_metrics", config_path=config, runner=run_metrics)


@evaluate_app.command("delong")
def evaluate_delong(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="evaluate_delong", config_path=config, runner=run_delong)


@evaluate_app.command("dca")
def evaluate_dca(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="evaluate_dca", config_path=config, runner=run_dca)


@explain_app.command("shap")
def explain_shap(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="explain_shap", config_path=config, runner=run_shap)


@report_app.command("consort")
def report_consort(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="report_consort", config_path=config, runner=run_consort)


@report_app.command("tables")
def report_tables(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="report_tables", config_path=config, runner=run_tables)


@report_app.command("curves")
def report_curves(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="report_curves", config_path=config, runner=run_curves)


@report_app.command("manuscript")
def report_manuscript(config: str | None = typer.Option(None, "--config")) -> None:
    _run_command(stage_name="report_manuscript", config_path=config, runner=run_manuscript)


@compat_app.command("export-legacy")
def compat_export_legacy(config: str | None = typer.Option(None, "--config")) -> None:
    cfg = _cfg(config)
    artifacts = ArtifactManager(cfg)
    _echo({"outputs": [str(path) for path in export_legacy_datasets(artifacts)]})


@runtime_app.command("inspect")
def runtime_inspect(config: str | None = typer.Option(None, "--config")) -> None:
    _echo(runtime_summary(_cfg(config)))


@runtime_app.command("benchmark")
def runtime_benchmark(
    config: str | None = typer.Option(None, "--config"),
    profiles: str = typer.Option("throughput,balanced", "--profiles"),
    targets: str = typer.Option("run_all", "--targets"),
    repeats: int = typer.Option(1, "--repeats", min=1),
    model_keys: str = typer.Option("", "--model-keys"),
    dataset_regimes: str = typer.Option("", "--dataset-regimes"),
    execution_policy: str = typer.Option("optimized_low_cpu", "--execution-policy"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
) -> None:
    cfg = _cfg(config)
    if output_dir is None:
        benchmark_dir = ArtifactManager(cfg).resolve("benchmarks")
    else:
        benchmark_dir = Path(output_dir)
        if not benchmark_dir.is_absolute():
            benchmark_dir = REPO_ROOT / benchmark_dir
    _echo(
        run_runtime_benchmarks(
            config_path=config,
            profiles=_parse_csv_values(profiles),
            targets=_parse_csv_values(targets),
            repeats=repeats,
            output_dir=benchmark_dir,
            model_keys=_parse_csv_values(model_keys) or None,
            dataset_regimes=_parse_csv_values(dataset_regimes) or None,
            execution_policy=execution_policy,
        )
    )


@run_app.command("all")
def run_all(config: str | None = typer.Option(None, "--config")) -> None:
    cfg = _cfg(config)
    progress = _progress_logger(cfg, stdout=True)
    start = perf_counter()
    if progress is not None:
        progress.emit_event(event_type="run_start", stage="run_all", status="started", message="orchestrated pipeline started")
    stage_order = [
        ("preprocess_preop", run_preop),
        ("preprocess_intraop", run_intraop),
        ("preprocess_tabular", run_tabular),
        ("preprocess_labels", run_labels),
        ("preprocess_timeseries", run_timeseries),
        ("preprocess_sequence", run_sequence),
        ("tune_tabular", run_tune_tabular),
    ]
    outputs: dict[str, Any] = {}
    try:
        for stage_name, runner in stage_order:
            outputs[stage_name] = _run_stage(stage_name=stage_name, runner=runner, config=cfg, progress=progress)

        orchestration_mode = cfg.get("runtime", {}).get("orchestration", {}).get("mode", "serial")
        if orchestration_mode == "overlap":
            outputs.update(_run_overlap_branch(config=cfg, config_path=config, progress=progress))
        else:
            outputs["tune_sequence"] = _run_stage(
                stage_name="tune_sequence",
                runner=run_tune_sequence,
                config=cfg,
                progress=progress,
            )
            outputs["train_tabular"] = _run_stage(
                stage_name="train_tabular",
                runner=run_train_tabular,
                config=cfg,
                progress=progress,
            )

        tail_stages = [
            ("train_sequence", run_train_sequence),
            ("evaluate_calibrate", run_calibration),
            ("evaluate_metrics", run_metrics),
            ("evaluate_delong", run_delong),
            ("evaluate_dca", run_dca),
            ("report_manuscript", run_manuscript),
        ]
        for stage_name, runner in tail_stages:
            outputs[stage_name] = _run_stage(stage_name=stage_name, runner=runner, config=cfg, progress=progress)
    except typer.Exit as exc:
        exit_code = getattr(exc, "exit_code", 1)
        if progress is not None:
            progress.emit_event(
                event_type="run_end",
                stage="run_all",
                status="aborted" if exit_code == 130 else "error",
                message="orchestrated pipeline interrupted" if exit_code == 130 else "orchestrated pipeline failed",
                wall_time_seconds=perf_counter() - start,
            )
        raise
    except KeyboardInterrupt as exc:
        if progress is not None:
            progress.emit_event(
                event_type="run_end",
                stage="run_all",
                status="aborted",
                message="orchestrated pipeline interrupted",
                wall_time_seconds=perf_counter() - start,
            )
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        if progress is not None:
            progress.emit_event(
                event_type="run_end",
                stage="run_all",
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                wall_time_seconds=perf_counter() - start,
            )
        raise
    if progress is not None:
        progress.emit_event(
            event_type="run_end",
            stage="run_all",
            status="completed",
            message="orchestrated pipeline finished",
            wall_time_seconds=perf_counter() - start,
        )
    _echo(outputs)


def main() -> None:
    app()
