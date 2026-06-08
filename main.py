import logging

import typer
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from sara_rain_drop_detector.cli_inputs import (
    parse_input_blob_storage_locations,
)
from sara_rain_drop_detector.config.logger import setup_logger
from sara_rain_drop_detector.config.open_telemetry import setup_open_telemetry
from sara_rain_drop_detector.config.settings import settings
from sara_rain_drop_detector.main_rain_drop_detector import (
    run_rain_drop_detection_workflow,
)

setup_logger()
logger = logging.getLogger(__name__)
setup_open_telemetry()
tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)


app = typer.Typer()


@app.command()
def run_rain_drop_detection(
    input_blob_storage_locations: str = typer.Option(
        ...,
        help=(
            "JSON array of blob locations to analyze. sara-rain-drop-detector "
            "is a single-image analyzer, so the array must contain exactly "
            "one entry."
        ),
    ),
    result_output_file: str = typer.Option(
        "/tmp/result.json",
        help=(
            "Path to write the workflow result JSON to. The file is created "
            "after the workflow succeeds and contains 'rain' (bool)."
        ),
    ),
) -> None:
    image_location = parse_input_blob_storage_locations(input_blob_storage_locations)[0]

    with tracer.start_as_current_span(
        "cli.run",
        attributes={
            "src.container": image_location.blob_container,
            "src.blob": image_location.blob_name,
        },
    ) as span:
        try:
            run_rain_drop_detection_workflow(
                image_location,
                result_output_file,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise


if __name__ == "__main__":
    app()
