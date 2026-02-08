"""CLI entry-point for the processing pipeline."""

from __future__ import annotations

import logging

import click

from packages.pipeline.process import process_scan_to_json


@click.group()
def main():
    """Digital Twin point-cloud processing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_file", default=None, help="Output JSON path.")
@click.option("--voxel-size", default=0.05, show_default=True, help="Voxel grid size (metres).")
@click.option("--max-planes", default=10, show_default=True, help="Max planes to detect.")
@click.option("--seed", default=42, show_default=True, help="RANSAC random seed.")
def process(
    input_file: str, output_file: str | None, voxel_size: float, max_planes: int, seed: int,
):
    """Process a point-cloud file and produce a parametric model JSON."""
    json_str = process_scan_to_json(
        input_file,
        output_path=output_file,
        voxel_size=voxel_size,
        max_planes=max_planes,
        seed=seed,
    )
    click.echo(json_str)


if __name__ == "__main__":
    main()
