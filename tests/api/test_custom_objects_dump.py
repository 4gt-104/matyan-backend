"""Unit tests for custom object _dump_value methods, _downsample, _fetch_blob_from_s3."""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from matyan_backend.api.runs._custom_objects import (
    AudioApiConfig,
    DistributionApiConfig,
    FigureApiConfig,
    ImageApiConfig,
    TextApiConfig,
    _downsample,
    _fetch_blob_from_s3,
)
from matyan_backend.storage import runs, sequences

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestDownsample:
    def test_fewer_than_density(self) -> None:
        items = [1, 2, 3]
        assert _downsample(items, 10) == items

    def test_exact_density(self) -> None:
        items = [1, 2, 3]
        assert _downsample(items, 3) == items

    def test_density_one(self) -> None:
        assert _downsample([1, 2, 3, 4, 5], 1) == [1]

    def test_density_zero(self) -> None:
        items = [1, 2, 3]
        assert _downsample(items, 0) == items

    def test_downsample_picks_evenly(self) -> None:
        items = list(range(10))
        result = _downsample(items, 3)
        assert len(result) == 3
        assert result[0] == 0
        assert result[-1] == 9


class TestImageDumpValue:
    def test_dict_input(self) -> None:
        result = ImageApiConfig._dump_value(  # noqa: SLF001
            {"caption": "cat", "width": 100, "height": 200, "format": "jpeg"},
            "run1",
            0,
            "images",
            5,
            0,
        )
        assert result["caption"] == "cat"
        assert result["width"] == 100
        assert result["height"] == 200
        assert result["format"] == "jpeg"
        assert "blob_uri" in result
        assert result["index"] == 0

    def test_dict_input_default_format(self) -> None:
        result = ImageApiConfig._dump_value(  # noqa: SLF001
            {"caption": "cat", "width": 100, "height": 200},
            "run1",
            0,
            "images",
            5,
            0,
        )
        assert result["format"] == "png"

    def test_non_dict_input(self) -> None:
        result = ImageApiConfig._dump_value("raw", "run1", 0, "images", 0, 2)  # noqa: SLF001
        assert result["caption"] == ""
        assert result["width"] == 0
        assert result["format"] == "png"
        assert result["index"] == 2
        assert "blob_uri" in result


class TestTextDumpValue:
    def test_dict_input(self) -> None:
        result = TextApiConfig._dump_value({"data": "Hello world"}, "r", 0, "texts", 0, 0)  # noqa: SLF001
        assert result["data"] == "Hello world"
        assert result["index"] == 0

    def test_non_dict_string(self) -> None:
        result = TextApiConfig._dump_value("plain text", "r", 0, "texts", 0, 0)  # noqa: SLF001
        assert result["data"] == "plain text"

    def test_non_dict_none(self) -> None:
        result = TextApiConfig._dump_value(None, "r", 0, "texts", 0, 0)  # noqa: SLF001
        assert result["data"] == ""


class TestDistributionDumpValue:
    def test_dict_with_bytes_blob(self) -> None:
        blob = struct.pack("<3d", 1.0, 2.0, 3.0)
        result = DistributionApiConfig._dump_value(  # noqa: SLF001
            {"data": blob, "bin_count": 3, "range": [0, 10]},
            "r",
            0,
            "distributions",
            0,
            0,
        )
        assert result["data"]["type"] == "numpy"
        assert result["data"]["shape"] == 3
        assert result["data"]["dtype"] == "float64"
        assert result["bin_count"] == 3
        assert result["range"] == [0, 10]

    def test_dict_with_list_blob(self) -> None:
        result = DistributionApiConfig._dump_value(  # noqa: SLF001
            {"data": [1.0, 2.0], "bin_count": 2, "range": [0, 5]},
            "r",
            0,
            "distributions",
            0,
            0,
        )
        assert result["data"]["shape"] == 2
        expected_blob = struct.pack("<2d", 1.0, 2.0)
        assert result["data"]["blob"] == expected_blob

    def test_dict_with_unsupported_blob_type(self) -> None:
        result = DistributionApiConfig._dump_value(  # noqa: SLF001
            {"data": 42, "bin_count": 0},
            "r",
            0,
            "distributions",
            0,
            0,
        )
        assert result["data"]["blob"] == b""

    def test_non_dict(self) -> None:
        result = DistributionApiConfig._dump_value(None, "r", 0, "distributions", 0, 0)  # noqa: SLF001
        assert result["data"]["shape"] == 0
        assert result["bin_count"] == 0


class TestAudioDumpValue:
    def test_dict_input(self) -> None:
        result = AudioApiConfig._dump_value(  # noqa: SLF001
            {"caption": "sound", "format": "mp3"},
            "r",
            0,
            "audios",
            1,
            0,
        )
        assert result["caption"] == "sound"
        assert result["format"] == "mp3"
        assert "blob_uri" in result

    def test_non_dict(self) -> None:
        result = AudioApiConfig._dump_value("raw", "r", 0, "audios", 0, 0)  # noqa: SLF001
        assert result["format"] == "wav"
        assert result["caption"] == ""


class TestFetchBlobFromS3:
    def test_no_values(self, db: Database) -> None:
        runs.create_run(db, "fb1")
        result = _fetch_blob_from_s3(db, "fb1", 0, "images", 999)
        assert result == b""

    def test_non_dict_value(self, db: Database) -> None:
        runs.create_run(db, "fb2")
        sequences.write_sequence_step(db, "fb2", 0, "images", 0, "not-a-dict")
        result = _fetch_blob_from_s3(db, "fb2", 0, "images", 0)
        assert result == b""

    def test_no_s3_key(self, db: Database) -> None:
        runs.create_run(db, "fb3")
        sequences.write_sequence_step(db, "fb3", 0, "images", 0, {"data": "no-s3-key"})
        result = _fetch_blob_from_s3(db, "fb3", 0, "images", 0)
        assert result == b""

    @patch("matyan_backend.api.runs._custom_objects.get_blob", return_value=b"blob-content")
    def test_success(self, mock_get: MagicMock, db: Database) -> None:  # noqa: ARG002
        runs.create_run(db, "fb4")
        sequences.write_sequence_step(db, "fb4", 0, "images", 0, {"s3_key": "fb4/img.png"})
        result = _fetch_blob_from_s3(db, "fb4", 0, "images", 0)
        assert result == b"blob-content"

    @patch("matyan_backend.api.runs._custom_objects.get_blob", return_value=b"img-bytes")
    def test_list_valued_step_index_zero(self, mock_get: MagicMock, db: Database) -> None:  # noqa: ARG002
        """Images are stored as lists; index selects within the list."""
        runs.create_run(db, "fb5")
        step_val = [{"s3_key": "fb5/img0.png"}, {"s3_key": "fb5/img1.png"}]
        sequences.write_sequence_step(db, "fb5", 0, "images", 0, step_val)
        result = _fetch_blob_from_s3(db, "fb5", 0, "images", 0, index=0)
        assert result == b"img-bytes"

    @patch("matyan_backend.api.runs._custom_objects.get_blob", return_value=b"second-img")
    def test_list_valued_step_index_one(self, mock_get: MagicMock, db: Database) -> None:  # noqa: ARG002
        runs.create_run(db, "fb6")
        step_val = [{"s3_key": "fb6/img0.png"}, {"s3_key": "fb6/img1.png"}]
        sequences.write_sequence_step(db, "fb6", 0, "images", 0, step_val)
        result = _fetch_blob_from_s3(db, "fb6", 0, "images", 0, index=1)
        assert result == b"second-img"

    def test_list_valued_step_index_out_of_range(self, db: Database) -> None:
        runs.create_run(db, "fb7")
        step_val = [{"s3_key": "fb7/img0.png"}]
        sequences.write_sequence_step(db, "fb7", 0, "images", 0, step_val)
        result = _fetch_blob_from_s3(db, "fb7", 0, "images", 0, index=5)
        assert result == b""


class TestFigureDumpValue:
    def test_dict_with_dict_data(self) -> None:
        fig_data = {"data": [{"x": [1, 2], "y": [3, 4]}], "layout": {}}
        result = FigureApiConfig._dump_value(  # noqa: SLF001
            {"data": fig_data},
            "r",
            0,
            "figures",
            0,
            0,
        )
        parsed = json.loads(result["data"])
        assert "data" in parsed

    def test_dict_with_string_data(self) -> None:
        result = FigureApiConfig._dump_value(  # noqa: SLF001
            {"data": "already-json"},
            "r",
            0,
            "figures",
            0,
            0,
        )
        assert result["data"] == "already-json"

    def test_non_dict(self) -> None:
        result = FigureApiConfig._dump_value(None, "r", 0, "figures", 0, 0)  # noqa: SLF001
        assert result["data"] == "{}"
