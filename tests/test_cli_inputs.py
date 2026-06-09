import pytest
import typer

from sara_rain_drop_detector.cli_inputs import parse_input_blob_storage_locations


def test_parses_single_entry_camel_case() -> None:
    raw = '[{"blobContainer": "c", "blobName": "n"}]'

    result = parse_input_blob_storage_locations(raw)

    assert len(result) == 1
    assert result[0].blob_container == "c"
    assert result[0].blob_name == "n"


def test_rejects_invalid_json() -> None:
    with pytest.raises(typer.BadParameter, match="not valid JSON"):
        parse_input_blob_storage_locations("not-json")


def test_rejects_multiple_entries() -> None:
    raw = (
        '[{"blobContainer": "c", "blobName": "n1"},'
        '{"blobContainer": "c", "blobName": "n2"}]'
    )

    with pytest.raises(typer.BadParameter, match="exactly one entry"):
        parse_input_blob_storage_locations(raw)
