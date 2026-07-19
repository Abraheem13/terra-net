"""Typer CLI entry point: `terranet <command>`."""
from __future__ import annotations

import typer

app = typer.Typer(help="TERRA-Net experiment CLI")


@app.command()
def info():
    """Print environment and GPU info."""
    import torch

    typer.echo(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(torch.cuda.get_device_name(0))


@app.command()
def version():
    from terranet import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
