"""
CLI Tool - Command-line interface for AI SimTest.
Run simulations directly from the terminal.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AI SimTest - Open-source AI simulation testing platform."""
    pass


@main.command()
@click.option("--bot-endpoint", required=True, help="API endpoint of the bot to test")
@click.option("--bot-api-key", default=None, help="API key for the bot")
@click.option("--bot-format", default="openai", help="Request format: openai, anthropic, custom")
@click.option("--documentation", default="", help="Path to documentation file or inline text")
@click.option("--personas", default=20, help="Number of personas to generate")
@click.option("--max-turns", default=15, help="Max conversation turns")
@click.option("--parallel", default=10, help="Max parallel conversations")
@click.option("--output", default="./reports", help="Output directory for reports")
@click.option("--export-formats", default="jsonl,csv,summary", help="Comma-separated export formats")
@click.option("--config-file", default=None, help="Path to YAML/JSON config file")
def run(
    bot_endpoint: str,
    bot_api_key: str | None,
    bot_format: str,
    documentation: str,
    personas: int,
    max_turns: int,
    parallel: int,
    output: str,
    export_formats: str,
    config_file: str | None,
):
    """Run a simulation test against your AI chatbot."""
    from src.core.logging import setup_logging
    setup_logging()

    console.print(Panel.fit(
        "[bold blue]AI SimTest[/] - Simulation Testing Platform",
        subtitle="v0.1.0",
    ))

    # Load documentation from file if path provided
    doc_text = documentation
    if documentation and Path(documentation).exists():
        doc_text = Path(documentation).read_text()

    # Load config from file if provided
    if config_file:
        config_data = json.loads(Path(config_file).read_text())
        # Merge CLI args with config file (CLI takes precedence)
        bot_endpoint = bot_endpoint or config_data.get("bot_endpoint", "")
        doc_text = doc_text or config_data.get("documentation", "")
        personas = personas or config_data.get("num_personas", 20)

    asyncio.run(_run_simulation(
        bot_endpoint=bot_endpoint,
        bot_api_key=bot_api_key,
        bot_format=bot_format,
        documentation=doc_text,
        num_personas=personas,
        max_turns=max_turns,
        max_parallel=parallel,
        output_dir=output,
        export_formats=export_formats.split(","),
    ))


async def _run_simulation(
    bot_endpoint: str,
    bot_api_key: str | None,
    bot_format: str,
    documentation: str,
    num_personas: int,
    max_turns: int,
    max_parallel: int,
    output_dir: str,
    export_formats: list[str],
):
    """Async simulation runner."""
    from src.core.orchestrator import SimulationOrchestrator
    from src.models import BotConfig, SimulationConfig

    config = SimulationConfig(
        name="CLI Simulation Run",
        bot=BotConfig(
            api_endpoint=bot_endpoint,
            api_key=bot_api_key,
            request_format=bot_format,
        ),
        documentation=documentation,
        num_personas=num_personas,
        max_turns_per_conversation=max_turns,
        max_parallel_conversations=max_parallel,
    )

    orchestrator = SimulationOrchestrator(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running simulation...", total=None)

        report = await orchestrator.run_simulation()

        progress.update(task, description="Exporting results...")
        exported = orchestrator.export_results(output_dir=output_dir, formats=export_formats)

    # Display results
    _print_report(report, exported)


def _print_report(report, exported: dict[str, str]):
    """Print a formatted report to the console."""
    from src.models import SimulationReport

    s = report.summary

    # Summary table
    table = Table(title="Simulation Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Personas", str(s.total_personas))
    table.add_row("Total Conversations", str(s.total_conversations))
    table.add_row("Total Turns", str(s.total_turns))
    table.add_row("Pass Rate", f"{s.pass_rate:.1%}")
    table.add_row("Average Score", f"{s.average_score:.2f}")
    table.add_row("Critical Failures", str(s.critical_failures))
    table.add_row("Warnings", str(s.warnings))
    table.add_row("Execution Time", f"{s.execution_time_seconds:.1f}s")

    console.print(table)

    # Judge scores
    if report.score_by_judge:
        console.print("\n[bold]Scores by Judge:[/]")
        for judge, score in report.score_by_judge.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
            console.print(f"  {judge:20s} [{color}]{bar}[/] {score:.1%}")

    # Recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/]")
        for rec in report.recommendations:
            console.print(f"  {rec}")

    # Exported files
    if exported:
        console.print("\n[bold]Exported Files:[/]")
        for fmt, path in exported.items():
            console.print(f"  {fmt}: {path}")

    console.print()


@main.command()
def serve():
    """Start the API server."""
    import uvicorn
    from src.core.config import settings

    console.print(Panel.fit(
        f"[bold blue]AI SimTest API Server[/]\nhttp://{settings.api_host}:{settings.api_port}",
    ))

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )


@main.command()
def version():
    """Show version info."""
    console.print("[bold]AI SimTest[/] v0.1.0")
    console.print("Open-source AI simulation testing platform")


if __name__ == "__main__":
    main()
