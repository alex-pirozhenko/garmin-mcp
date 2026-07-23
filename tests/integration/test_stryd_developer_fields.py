"""
Tests for Stryd/ConnectIQ developer-field power parsing and the
get_activity_decoupling tool.

Two styles are used:
  - Mock-based unit tests (matching the existing test_activity_analysis_tools.py
    convention) for targeted field-merging/gating behavior.
  - A real end-to-end test against tests/fixtures/stryd_run.fit — a synthetic
    but structurally real FIT file (built with fit-tool, see
    tests/fixtures/generate_stryd_fixture.py) parsed by the *actual* fitparse
    library, not mocked. This is the closest available substitute for the
    real reference activity (Garmin 23690795451 / TrainingPeaks 3747106892)
    described in the task, since raw FIT bytes for that activity aren't
    reachable from this environment (Garmin's download tools save
    server-side; see the PR description's manual verification checklist for
    how this was cross-checked against the live activity instead).
"""
import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from mcp.server.fastmcp import FastMCP

from garmin_mcp import activity_analysis
from garmin_mcp.activity_analysis import _compute_normalized_power, _compute_decoupling

ACTIVITY_ID = 23690795451
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "stryd_run.fit")


@pytest.fixture
def app_with_activity_analysis(mock_garmin_client):
    activity_analysis.configure(mock_garmin_client)
    app = FastMCP("Test Activity Analysis")
    app = activity_analysis.register_tools(app)
    return app


def _make_mock_fit_message(name, fields: dict):
    msg = Mock()
    msg.name = name
    msg.get_value = lambda field, *args: fields.get(field)
    return msg


def _mock_fitfile(messages):
    mock_ff = MagicMock()
    mock_ff.get_messages.return_value = iter(messages)
    return mock_ff


# ---------------------------------------------------------------------------
# Normalized Power helper (unit tests)
# ---------------------------------------------------------------------------

def test_normalized_power_constant_equals_average():
    """NP of a perfectly constant power stream equals the average."""
    powers = [200] * 60
    assert _compute_normalized_power(powers) == pytest.approx(200, abs=0.01)


def test_normalized_power_variable_exceeds_average():
    """NP of a variable stream is higher than the simple average (4th-power
    weighting penalizes spikes)."""
    powers = [100] * 30 + [400] * 30
    np_w = _compute_normalized_power(powers)
    avg = sum(powers) / len(powers)
    assert np_w > avg


def test_normalized_power_short_series_falls_back():
    """Series shorter than the 30s window still produces a value."""
    assert _compute_normalized_power([200, 210, 190]) is not None


# ---------------------------------------------------------------------------
# Developer field discovery (mock-based)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stryd_power_used_when_no_native_power(app_with_activity_analysis, mock_garmin_client):
    """Session avg power/NP are backfilled from the Stryd developer field
    when there's no native power-meter aggregate (running w/ Stryd footpod)."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20

    field_desc = _make_mock_fit_message("field_description", {"field_name": "Power"})
    session_msg = _make_mock_fit_message("session", {"sport": "running"})
    records = [
        _make_mock_fit_message("record", {"heart_rate": 150, "Power": p})
        for p in [270, 275, 280, 272, 278] * 20  # 100 records
    ]

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile([field_desc, session_msg] + records)
        result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
        )

    data = json.loads(result[0][0].text)
    assert data["developer_fields_found"] == ["Power"]
    assert data["session"]["avg_power_w"] == 275
    assert "normalized_power_w" in data["session"]
    assert "variability_index" in data["session"]


@pytest.mark.asyncio
async def test_native_power_takes_priority_over_dev_field(app_with_activity_analysis, mock_garmin_client):
    """When both a native power field and a Stryd dev field are present on
    the same record, the native value wins (real power meter > footpod)."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20

    field_desc = _make_mock_fit_message("field_description", {"field_name": "Power"})
    records = [
        _make_mock_fit_message("record", {"power": 300, "Power": 150, "heart_rate": 140})
        for _ in range(5)
    ]

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile([field_desc] + records)
        result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data",
            {"activity_id": ACTIVITY_ID, "include_records": True},
        )

    data = json.loads(result[0][0].text)
    assert data["records"][0]["power_w"] == 300
    assert data["records"][0]["power_source"] == "native"


@pytest.mark.asyncio
async def test_unrecognized_developer_field_still_discoverable(app_with_activity_analysis, mock_garmin_client):
    """A developer field we don't have a dedicated aggregate for still shows
    up in developer_fields_found (discovery-first, don't hardcode)."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20

    field_desc = _make_mock_fit_message("field_description", {"field_name": "Some Future Metric"})

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile([field_desc])
        result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
        )

    data = json.loads(result[0][0].text)
    assert data["developer_fields_found"] == ["Some Future Metric"]


@pytest.mark.asyncio
async def test_avg_cadence_spm_added_for_running(app_with_activity_analysis, mock_garmin_client):
    """avg_cadence_spm = 2x avg_cadence_rpm for running (per-leg rpm -> total spm)."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20
    session_msg = _make_mock_fit_message("session", {"sport": "running", "avg_cadence": 79})

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile([session_msg])
        result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
        )

    data = json.loads(result[0][0].text)
    assert data["session"]["avg_cadence_rpm"] == 79
    assert data["session"]["avg_cadence_spm"] == 158


@pytest.mark.asyncio
async def test_avg_cadence_spm_not_added_for_cycling(app_with_activity_analysis, mock_garmin_client):
    """Cycling cadence is genuine crank RPM and must not be doubled."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20
    session_msg = _make_mock_fit_message("session", {"sport": "cycling", "avg_cadence": 85})

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile([session_msg])
        result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
        )

    data = json.loads(result[0][0].text)
    assert data["session"]["avg_cadence_rpm"] == 85
    assert "avg_cadence_spm" not in data["session"]


@pytest.mark.asyncio
async def test_min_duration_minutes_threaded_to_hr_drift(app_with_activity_analysis, mock_garmin_client):
    """min_duration_minutes overrides the previous hard-coded 60min HR-drift floor."""
    mock_garmin_client.download_activity.return_value = b"\x00" * 20
    # ~17 minutes of data: below the default 60min floor, above a 10min floor.
    records = [
        _make_mock_fit_message("record", {"power": 200, "heart_rate": 145})
        for _ in range(1000)
    ]

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile(records)
        default_result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
        )

    with patch("garmin_mcp.activity_analysis.fitparse") as mock_fp:
        mock_fp.FitFile.return_value = _mock_fitfile(records)
        short_floor_result = await app_with_activity_analysis.call_tool(
            "get_activity_fit_data",
            {"activity_id": ACTIVITY_ID, "min_duration_minutes": 10},
        )

    default_data = json.loads(default_result[0][0].text)
    short_floor_data = json.loads(short_floor_result[0][0].text)
    assert "hr_drift" not in default_data["session"]
    assert "hr_drift" in short_floor_data["session"]


# ---------------------------------------------------------------------------
# download_activity_file: return_base64
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_activity_file_return_base64_no_dir_needed(
    app_with_activity_analysis, mock_garmin_client, monkeypatch, tmp_path
):
    """return_base64=True skips the needs_setup gate and returns file bytes
    inline for a client that can't reach the MCP server's filesystem."""
    monkeypatch.delenv("GARMIN_FIT_DOWNLOAD_DIR", raising=False)
    monkeypatch.setenv("GARMIN_FIT_CONFIG", str(tmp_path / "missing.json"))

    payload = b"<gpx>data</gpx>"
    mock_garmin_client.download_activity.return_value = payload

    result = await app_with_activity_analysis.call_tool(
        "download_activity_file",
        {"activity_id": ACTIVITY_ID, "format": "gpx", "return_base64": True},
    )
    data = json.loads(result[0][0].text)

    import base64
    assert base64.b64decode(data["content_base64"]) == payload
    assert "file_path" not in data


@pytest.mark.asyncio
async def test_download_activity_file_return_base64_over_cap(
    app_with_activity_analysis, mock_garmin_client, monkeypatch, tmp_path
):
    """Files over the 5MB cap get a base64_error instead of a huge payload."""
    monkeypatch.delenv("GARMIN_FIT_DOWNLOAD_DIR", raising=False)
    monkeypatch.setenv("GARMIN_FIT_CONFIG", str(tmp_path / "missing.json"))

    payload = b"x" * (5 * 1024 * 1024 + 1)
    mock_garmin_client.download_activity.return_value = payload

    result = await app_with_activity_analysis.call_tool(
        "download_activity_file",
        {"activity_id": ACTIVITY_ID, "format": "gpx", "return_base64": True},
    )
    data = json.loads(result[0][0].text)

    assert "content_base64" not in data
    assert "exceeds the 5 MB" in data["base64_error"]


@pytest.mark.asyncio
async def test_download_activity_file_saves_and_returns_base64_together(
    app_with_activity_analysis, mock_garmin_client, tmp_path
):
    """When both output_dir and return_base64 are given, the file is saved
    AND base64 is returned (additive, not either/or)."""
    payload = b"<gpx>data</gpx>"
    mock_garmin_client.download_activity.return_value = payload

    result = await app_with_activity_analysis.call_tool(
        "download_activity_file",
        {
            "activity_id": ACTIVITY_ID,
            "format": "gpx",
            "output_dir": str(tmp_path),
            "return_base64": True,
        },
    )
    data = json.loads(result[0][0].text)

    assert (tmp_path / f"{ACTIVITY_ID}.gpx").read_bytes() == payload
    assert "content_base64" in data
    assert data["file_path"] == str((tmp_path / f"{ACTIVITY_ID}.gpx").resolve())


# ---------------------------------------------------------------------------
# Real end-to-end test against a synthetic-but-real FIT binary
# ---------------------------------------------------------------------------

@pytest.fixture
def stryd_fit_bytes():
    with open(FIXTURE_PATH, "rb") as f:
        return f.read()


@pytest.mark.asyncio
async def test_real_fit_file_stryd_power_end_to_end(
    app_with_activity_analysis, mock_garmin_client, stryd_fit_bytes
):
    """Parses a real (synthetic) FIT binary with actual fitparse (not
    mocked) — session shows Stryd-derived power/NP, lap intensity gates the
    warmup lap correctly, and cadence is converted to spm for running."""
    mock_garmin_client.download_activity.return_value = stryd_fit_bytes

    result = await app_with_activity_analysis.call_tool(
        "get_activity_fit_data", {"activity_id": ACTIVITY_ID}
    )
    data = json.loads(result[0][0].text)

    assert data["session"]["sport"] == "running"
    assert set(data["developer_fields_found"]) >= {"Power", "Form Power", "Leg Spring Stiffness", "Air Power"}
    assert data["session"]["avg_cadence_spm"] == 158

    # Warmup laps are ~150W, active (work) lap is ~274W/274W NP by construction.
    active_lap = next(lap for lap in data["laps"] if lap.get("intensity") == "active")
    assert active_lap["avg_power_w"] == pytest.approx(274, abs=3)
    assert active_lap["normalized_power_w"] == pytest.approx(274, abs=5)
    assert "avg_form_power_w" in active_lap
    assert "avg_leg_spring_stiffness_kn_m" in active_lap


@pytest.mark.asyncio
async def test_real_fit_file_decoupling_grade_confounded(
    app_with_activity_analysis, mock_garmin_client, stryd_fit_bytes
):
    """get_activity_decoupling on the synthetic climb-then-descend route
    correctly flags grade_confounded and stays well under the 2KB response cap."""
    mock_garmin_client.download_activity.return_value = stryd_fit_bytes

    result = await app_with_activity_analysis.call_tool(
        "get_activity_decoupling", {"activity_id": ACTIVITY_ID}
    )
    text = result[0][0].text
    data = json.loads(text)

    assert data["method"] == "developer_field_power"
    assert data["grade_confounded"] is True
    assert data["net_elevation_first_half_m"] > 0
    assert data["net_elevation_second_half_m"] < 0
    assert "avg_cadence_first_half_spm" in data
    assert len(text.encode("utf-8")) < 2048


@pytest.mark.asyncio
async def test_real_fit_file_decoupling_excludes_warmup(
    app_with_activity_analysis, mock_garmin_client, stryd_fit_bytes
):
    """The warmup lap (300s) is excluded from the work portion: first+second
    half durations should sum to the 2700s active lap, not the full 3000s."""
    mock_garmin_client.download_activity.return_value = stryd_fit_bytes

    result = await app_with_activity_analysis.call_tool(
        "get_activity_decoupling", {"activity_id": ACTIVITY_ID}
    )
    data = json.loads(result[0][0].text)

    total = data["first_half"]["duration_s"] + data["second_half"]["duration_s"]
    assert total == 2700


@pytest.mark.asyncio
async def test_real_fit_file_decoupling_insufficient_data(
    app_with_activity_analysis, mock_garmin_client, stryd_fit_bytes
):
    """min_duration_minutes above the work portion's actual length (45 min) errors cleanly."""
    mock_garmin_client.download_activity.return_value = stryd_fit_bytes

    result = await app_with_activity_analysis.call_tool(
        "get_activity_decoupling",
        {"activity_id": ACTIVITY_ID, "min_duration_minutes": 50},
    )
    data = json.loads(result[0][0].text)

    assert data["error"] == "insufficient_data"
