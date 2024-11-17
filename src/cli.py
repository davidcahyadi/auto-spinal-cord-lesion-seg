import sys
from pathlib import Path

import click

from src.analyzer.image_processor import ImageProcessor
from src.utils.validators import validate_input_path


@click.command(help="MRI Scan Series Analyzer using Neural Networks")
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing MRI scan series",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./output",
    help="Output directory for analysis results (default: ./output)",
)
@click.option(
    "-w",
    "--weights",
    type=click.Path(exists=True),
    default="./weights",
    help="Weights directory (default: ./weights)",
)
def main(input, output, weights):
    # Validate input path
    input_path = Path(input)
    if not validate_input_path(input_path):
        sys.exit(1)

    # Initialize processor
    processor = ImageProcessor(weights_path=Path(weights), format="png")

    # Process the directory
    try:
        processor.process_directory(input_path, Path(output))
    except Exception as e:
        click.echo(f"Error processing directory: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
