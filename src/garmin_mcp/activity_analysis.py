"""
Activity analysis via FIT file parsing for Garmin Connect MCP Server

Exposes data not available through the REST API:
- DI2 / electronic shifting events (gear combinations, cadence at shift, shift quality, terrain)
- Advanced cycling dynamics (platform center offset, power phase, left/right balance per record)
- Full per-second time series (power, cadence, HR, speed, altitude, GPS)
- Power Duration Curve (best mean maximal power at key durations)
- Climb detection with VAM, grade analysis, W/kg
- HR drift / cardiac drift (aerobic decoupling)
- Temperature correlation with HR and power
- Variability Index per session and lap
"""
import base64
import datetime as _dt
import gzip
import io
import json
import os
import zipfile
from typing import Any, Dict, List, Optional, Union

try:
    import fitparse
    FITPARSE_AVAILABLE = True
except ImportError:
    FITPARSE_AVAILABLE = False

# The garmin_client will be set by the main file
garmin_client = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


# ---------------------------------------------------------------------------
# FIT decoding helpers
# ---------------------------------------------------------------------------

def _decode_gear_change(data: int) -> dict:
    """Decode a packed gear_change_data uint32 from a Di2 shift event.

    Shimano Di2 packs gear information as:
      bits 0-7:   rear gear number (1 = smallest/hardest cog)
      bits 8-15:  front gear number (1 = inner/small ring)
      bits 16-23: rear gear teeth count
      bits 24-31: front gear teeth count
    """
    rear_gear_num = data & 0xFF
    front_gear_num = (data >> 8) & 0xFF
    rear_teeth = (data >> 16) & 0xFF
    front_teeth = (data >> 24) & 0xFF
    return {
        "rear_gear_num": rear_gear_num,
        "front_gear_num": front_gear_num,
        "rear_teeth": rear_teeth if rear_teeth > 0 else None,
        "front_teeth": front_teeth if front_teeth > 0 else None,
    }


def _decode_left_right_balance(value) -> Optional[float]:
    """Decode Garmin's left_right_balance field to left power percentage."""
    if value is None:
        return None
    try:
        int_val = int(value)
        right_dominant = bool(int_val & 0x8000)
        pct = (int_val & 0x7FFF) / 100.0
        if right_dominant:
            return round(100.0 - pct, 1)
        return round(pct, 1)
    except (TypeError, ValueError):
        return None


def _get_field(message, *names):
    """Get the first matching field value from a FIT message."""
    for name in names:
        field = message.get_value(name)
        if field is not None:
            return field
    return None


def _semicircles_to_degrees(value) -> Optional[float]:
    """Convert FIT semicircle coordinates to decimal degrees."""
    if value is None:
        return None
    return round(value * (180.0 / 2**31), 6)


# ---------------------------------------------------------------------------
# ConnectIQ developer field discovery (Stryd running power etc.)
# ---------------------------------------------------------------------------
#
# Stryd's ConnectIQ data field writes running power (and derived metrics)
# into FIT "developer fields" rather than the native power record field,
# since the watch has no built-in power meter. Developer fields are declared
# by 'field_description' messages (name, base type, units) that precede the
# 'record' messages using them, keyed by a name we can query with the FIT
# SDK's message.get_value(name). We match names case-insensitively since
# ConnectIQ field names vary slightly across Stryd firmware/app versions.

_STRYD_ALIAS_MAP = {
    "power": "power",
    "form power": "form_power",
    "leg spring stiffness": "leg_spring_stiffness",
    "air power": "air_power",
    "vertical oscillation": "vertical_oscillation",
    "ground time": "ground_time",
}


def _dev_value(message, dev_field_name_by_key: dict, key: str):
    """Look up a discovered developer field's value on a record message by our internal key."""
    name = dev_field_name_by_key.get(key)
    if not name:
        return None
    return _get_field(message, name)


def _parse_iso(s):
    """Parse a FIT-style timestamp string ("2026-05-15 02:27:08", optionally with tz) to datetime."""
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_fit_bytes(raw: bytes) -> bytes:
    """Extract raw FIT bytes from whatever Garmin's download endpoint returns.

    Garmin's ORIGINAL format download returns a ZIP archive containing one or
    more .fit files. Handle that, plus fall back for gzip and raw FIT.
    """
    if raw[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            fit_names = [n for n in zf.namelist() if n.lower().endswith('.fit')]
            if not fit_names:
                raise ValueError("ZIP archive contains no .fit files")
            return zf.read(fit_names[0])

    if raw[:2] == b'\x1f\x8b':
        return gzip.decompress(raw)

    return raw


# ---------------------------------------------------------------------------
# Download directory config (for download_activity_file / set_fit_download_dir)
# ---------------------------------------------------------------------------

_DEFAULT_FIT_CONFIG = "~/.garminconnect_fit_config.json"


def _get_fit_config_path() -> str:
    """Path to the JSON config that stores the default download directory."""
    return os.getenv("GARMIN_FIT_CONFIG") or _DEFAULT_FIT_CONFIG


def _read_fit_config() -> dict:
    """Read the FIT download config. Returns {} if missing or invalid."""
    path = os.path.expanduser(_get_fit_config_path())
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_fit_config(dir_path: str) -> None:
    """Persist the default download directory to the JSON config."""
    path = os.path.expanduser(_get_fit_config_path())
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cfg = _read_fit_config()
    cfg["download_dir"] = dir_path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _resolve_download_dir(output_dir: Optional[str]) -> Optional[str]:
    """Resolve the directory for saving activity files (first match wins):

    1. output_dir argument (one-off; not persisted)
    2. GARMIN_FIT_DOWNLOAD_DIR environment variable
    3. persisted config (download_dir)
    Returns an absolute path, or None when nothing is configured.
    """
    if output_dir:
        return os.path.abspath(os.path.expanduser(output_dir))
    env_dir = os.getenv("GARMIN_FIT_DOWNLOAD_DIR")
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    cfg_dir = _read_fit_config().get("download_dir")
    if cfg_dir:
        return os.path.abspath(os.path.expanduser(cfg_dir))
    return None


def _safe_avg(values: list) -> Optional[float]:
    """Return mean of a list of numbers, or None if empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe_round(value, ndigits: int = 1) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


# ---------------------------------------------------------------------------
# Normalized Power (30s rolling average, 4th-power mean) + power aggregates
# ---------------------------------------------------------------------------

def _compute_normalized_power(power_values: List[Optional[float]], window_s: int = 30) -> Optional[float]:
    """Standard (Coggan) Normalized Power: 30s rolling avg, raised to the 4th
    power, meaned, then 4th-rooted. Gaps are treated as 0 W (coasting/walking),
    matching the convention used by _compute_power_duration_curve.

    Falls back to a whole-series window when there's less than window_s of data.
    """
    vals = [v if v is not None else 0 for v in power_values]
    n = len(vals)
    if n == 0:
        return None
    w = min(window_s, n)

    window_sum = sum(vals[:w])
    rolling = [window_sum / w]
    for i in range(w, n):
        window_sum += vals[i] - vals[i - w]
        rolling.append(window_sum / w)

    if not rolling:
        return None
    fourth_power_mean = sum(r ** 4 for r in rolling) / len(rolling)
    return fourth_power_mean ** 0.25


def _compute_power_aggregates_for_range(
    records: List[Dict], start_ts=None, end_ts=None
) -> Dict[str, Any]:
    """Compute avg/max/NP/VI/EF/form-power/leg-spring-stiffness from per-second
    records, optionally restricted to a [start_ts, end_ts] datetime window.

    Used to backfill session/lap power summaries when power comes from a
    developer field (e.g. Stryd) rather than a native power-meter aggregate
    the device would normally precompute itself.
    """
    if start_ts is not None or end_ts is not None:
        sel = []
        for r in records:
            ts = _parse_iso(r.get("timestamp"))
            if ts is None:
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            sel.append(r)
    else:
        sel = records

    powers = [r["power_w"] for r in sel if r.get("power_w") is not None]
    if not powers:
        return {}

    out: Dict[str, Any] = {
        "avg_power_w": round(_safe_avg(powers)),
        "max_power_w": round(max(powers)),
    }

    np_w = _compute_normalized_power(powers)
    if np_w:
        out["normalized_power_w"] = round(np_w)
        if out["avg_power_w"]:
            out["variability_index"] = round(np_w / out["avg_power_w"], 3)
        hrs = [r["heart_rate_bpm"] for r in sel if r.get("heart_rate_bpm") is not None]
        avg_hr = _safe_avg(hrs)
        if avg_hr:
            out["ef"] = round(np_w / avg_hr, 2)

    form_powers = [r["form_power_w"] for r in sel if r.get("form_power_w") is not None]
    if form_powers:
        out["avg_form_power_w"] = round(_safe_avg(form_powers))

    lss_vals = [r["leg_spring_stiffness_kn_m"] for r in sel if r.get("leg_spring_stiffness_kn_m") is not None]
    if lss_vals:
        out["avg_leg_spring_stiffness_kn_m"] = round(_safe_avg(lss_vals), 2)

    return out


# ---------------------------------------------------------------------------
# Power Duration Curve (Mean Maximal Power)
# ---------------------------------------------------------------------------

_PDC_DURATIONS_S = [5, 30, 60, 300, 600, 1200, 3600]
_PDC_LABELS = ["5s", "30s", "1min", "5min", "10min", "20min", "60min"]


def _compute_power_duration_curve(records: List[Dict]) -> Optional[Dict]:
    """Compute best mean maximal power for standard durations.

    Uses a sliding window over the per-second power values.
    O(n * num_durations) — efficient for rides up to ~8 hours.
    """
    powers = [r.get("power_w") for r in records]
    if not any(p is not None for p in powers):
        return None

    # Fill None gaps with 0 (coasting) for window computation
    filled = [p if p is not None else 0 for p in powers]
    n = len(filled)
    if n == 0:
        return None

    # Precompute prefix sums for O(1) window queries
    prefix = [0] * (n + 1)
    for i, p in enumerate(filled):
        prefix[i + 1] = prefix[i] + p

    curve: Dict[str, Any] = {}
    for duration_s, label in zip(_PDC_DURATIONS_S, _PDC_LABELS):
        if n < duration_s:
            continue
        best = 0
        for i in range(n - duration_s + 1):
            window_sum = prefix[i + duration_s] - prefix[i]
            if window_sum > best:
                best = window_sum
        best_avg = best / duration_s
        if best_avg > 0:
            curve[label] = round(best_avg)

    return curve if curve else None


# ---------------------------------------------------------------------------
# Climb detection + VAM
# ---------------------------------------------------------------------------

def _detect_climbs(
    records: List[Dict],
    min_elevation_gain_m: float = 50.0,
    min_avg_grade_pct: float = 3.0,
    min_duration_s: int = 120,
) -> List[Dict]:
    """Segment the ride into climbs based on sustained positive grade.

    A climb starts when grade rises above min_avg_grade_pct and ends when it
    drops back below it for more than 30 seconds.
    """
    if not records:
        return []

    climbs = []
    in_climb = False
    climb_start_idx = 0
    flat_count = 0
    FLAT_TOLERANCE_S = 30

    for i, rec in enumerate(records):
        grade = rec.get("grade_pct")
        if grade is None:
            continue

        if not in_climb:
            if grade >= min_avg_grade_pct:
                in_climb = True
                climb_start_idx = i
                flat_count = 0
        else:
            if grade < min_avg_grade_pct:
                flat_count += 1
                if flat_count > FLAT_TOLERANCE_S:
                    # End the climb at where the flat started
                    end_idx = i - flat_count
                    climb = _summarize_climb(records, climb_start_idx, end_idx)
                    if (climb.get("elevation_gain_m", 0) >= min_elevation_gain_m and
                            climb.get("duration_s", 0) >= min_duration_s):
                        climbs.append(climb)
                    in_climb = False
                    flat_count = 0
            else:
                flat_count = 0

    # Close any open climb at end of ride
    if in_climb:
        climb = _summarize_climb(records, climb_start_idx, len(records) - 1)
        if (climb.get("elevation_gain_m", 0) >= min_elevation_gain_m and
                climb.get("duration_s", 0) >= min_duration_s):
            climbs.append(climb)

    return climbs


def _summarize_climb(records: List[Dict], start_idx: int, end_idx: int) -> Dict:
    """Compute aggregate stats for a climb segment."""
    segment = records[start_idx:end_idx + 1]
    if not segment:
        return {}

    duration_s = len(segment)  # 1 record ≈ 1 second
    start_alt = segment[0].get("altitude_m")
    end_alt = segment[-1].get("altitude_m")
    elevation_gain = (end_alt - start_alt) if (start_alt is not None and end_alt is not None) else None

    powers = [r["power_w"] for r in segment if r.get("power_w") is not None]
    cadences = [r["cadence_rpm"] for r in segment if r.get("cadence_rpm") is not None]
    hrs = [r["heart_rate_bpm"] for r in segment if r.get("heart_rate_bpm") is not None]
    grades = [r["grade_pct"] for r in segment if r.get("grade_pct") is not None]
    speeds = [r["speed_mps"] for r in segment if r.get("speed_mps") is not None]

    result: Dict[str, Any] = {
        "start_time": segment[0].get("timestamp", ""),
        "end_time": segment[-1].get("timestamp", ""),
        "duration_s": duration_s,
    }

    if elevation_gain is not None:
        result["elevation_gain_m"] = round(elevation_gain, 1)
        if duration_s > 0:
            result["vam_m_per_hr"] = round((elevation_gain / duration_s) * 3600)

    if speeds:
        total_dist = sum(s for s in speeds)  # m/s * 1s = m per record
        result["distance_m"] = round(total_dist)

    if grades:
        result["avg_grade_pct"] = round(_safe_avg(grades), 1)
        result["max_grade_pct"] = round(max(grades), 1)

    if powers:
        result["avg_power_w"] = round(_safe_avg(powers))

    if cadences:
        result["avg_cadence_rpm"] = round(_safe_avg(cadences))

    if hrs:
        result["avg_hr_bpm"] = round(_safe_avg(hrs))

    return result


# ---------------------------------------------------------------------------
# Grade analysis
# ---------------------------------------------------------------------------

def _grade_analysis(records: List[Dict]) -> Optional[Dict]:
    """Bin per-second records by grade and report avg power/cadence/HR per bin."""
    bins = {
        "descending": {"label": "descending (<-3%)", "records": []},
        "flat": {"label": "flat (-3% to 3%)", "records": []},
        "gentle": {"label": "gentle (3-6%)", "records": []},
        "moderate": {"label": "moderate (6-9%)", "records": []},
        "steep": {"label": "steep (>9%)", "records": []},
    }

    has_grade = False
    for rec in records:
        grade = rec.get("grade_pct")
        if grade is None:
            continue
        has_grade = True
        if grade < -3:
            bucket = "descending"
        elif grade < 3:
            bucket = "flat"
        elif grade < 6:
            bucket = "gentle"
        elif grade < 9:
            bucket = "moderate"
        else:
            bucket = "steep"
        bins[bucket]["records"].append(rec)

    if not has_grade:
        return None

    result = {}
    for key, data in bins.items():
        recs = data["records"]
        if not recs:
            continue
        entry: Dict[str, Any] = {"time_s": len(recs)}
        powers = [r["power_w"] for r in recs if r.get("power_w") is not None]
        cadences = [r["cadence_rpm"] for r in recs if r.get("cadence_rpm") is not None]
        hrs = [r["heart_rate_bpm"] for r in recs if r.get("heart_rate_bpm") is not None]
        if powers:
            entry["avg_power_w"] = round(_safe_avg(powers))
        if cadences:
            entry["avg_cadence_rpm"] = round(_safe_avg(cadences))
        if hrs:
            entry["avg_hr_bpm"] = round(_safe_avg(hrs))
        result[key] = entry

    return result if result else None


# ---------------------------------------------------------------------------
# HR drift / cardiac drift (aerobic decoupling)
# ---------------------------------------------------------------------------

def _compute_hr_drift(records: List[Dict], min_duration_minutes: float = 60) -> Optional[Dict]:
    """Compute aerobic decoupling (HR drift vs. power over the ride).

    Splits the ride into first and second halves by record count.
    Computes power:HR ratio for each half.
    Drift % = change in ratio from first to second half.
    Negative drift = HR increased relative to power (decoupling = less efficient).

    Args:
        min_duration_minutes: Minimum activity duration (assuming ~1 record/s)
            required before drift is computed (default 60, i.e. the previous
            hard-coded floor).
    """
    MIN_RECORDS = int(min_duration_minutes * 60)

    filtered = [
        r for r in records
        if r.get("power_w") is not None and r.get("heart_rate_bpm") is not None
        and r["heart_rate_bpm"] > 0
    ]

    if len(filtered) < MIN_RECORDS:
        return None

    mid = len(filtered) // 2
    first_half = filtered[:mid]
    second_half = filtered[mid:]

    def power_hr_ratio(recs):
        avg_p = _safe_avg([r["power_w"] for r in recs])
        avg_hr = _safe_avg([r["heart_rate_bpm"] for r in recs])
        if avg_p is None or avg_hr is None or avg_hr == 0:
            return None
        return avg_p / avg_hr

    r1 = power_hr_ratio(first_half)
    r2 = power_hr_ratio(second_half)

    if r1 is None or r2 is None or r1 == 0:
        return None

    drift_pct = ((r2 - r1) / r1) * 100

    if abs(drift_pct) < 5:
        interpretation = "well_coupled"
    elif abs(drift_pct) < 10:
        interpretation = "moderate_drift"
    else:
        interpretation = "significant_decoupling"

    return {
        "hr_drift_pct": round(drift_pct, 1),
        "first_half_power_hr_ratio": round(r1, 3),
        "second_half_power_hr_ratio": round(r2, 3),
        "interpretation": interpretation,
        "note": "Negative drift = HR increased vs power (decoupling). >10% suggests aerobic base is limiting factor.",
    }


# ---------------------------------------------------------------------------
# Pw:Hr / Pa:Hr decoupling (get_activity_decoupling)
# ---------------------------------------------------------------------------

def _work_portion_records(records: List[Dict], laps: List[Dict]) -> tuple:
    """Restrict records to the 'work' portion of the activity, excluding any
    laps whose FIT 'intensity' field is warmup/cooldown, when that data is
    present. Falls back to the whole activity otherwise.

    Returns (records_subset, window_start_dt_or_None, window_end_dt_or_None).
    """
    if not laps or not any(lap.get("intensity") in ("warmup", "cooldown") for lap in laps):
        return records, None, None

    work_laps = [lap for lap in laps if lap.get("intensity") not in ("warmup", "cooldown")]
    if not work_laps:
        return records, None, None

    start = _parse_iso(work_laps[0].get("start_time"))
    last_lap = work_laps[-1]
    last_start = _parse_iso(last_lap.get("start_time"))
    elapsed = last_lap.get("total_elapsed_time_s")
    if start is None or last_start is None or not elapsed:
        return records, None, None
    end = last_start + _dt.timedelta(seconds=float(elapsed))

    filtered = []
    for r in records:
        ts = _parse_iso(r.get("timestamp"))
        if ts is not None and start <= ts <= end:
            filtered.append(r)

    return (filtered if filtered else records), start, end


def _compute_decoupling(
    parsed: Dict[str, Any],
    split: Optional[float],
    min_duration_minutes: float,
) -> Dict[str, Any]:
    """Core computation behind get_activity_decoupling.

    Operates on an already fully-parsed FIT structure (session/laps/records,
    i.e. _parse_fit(..., include_records=True)) and returns a compact (<1KB)
    summary — never the underlying time series.
    """
    records: List[Dict] = parsed.get("records") or []
    laps: List[Dict] = parsed.get("laps") or []
    session: Dict[str, Any] = parsed.get("session") or {}
    is_running = str(session.get("sport", "")).lower() == "running"
    cadence_unit = "spm" if is_running else "rpm"
    cadence_mult = 2 if is_running else 1

    work_records, w_start, w_end = _work_portion_records(records, laps)

    min_records = int(min_duration_minutes * 60)
    if len(work_records) < min_records:
        return {
            "error": "insufficient_data",
            "message": (
                f"Work portion has {len(work_records)}s of data; need >= "
                f"{min_records}s ({min_duration_minutes} min)."
            ),
        }

    split_fraction = 0.5 if split is None else split
    split_fraction = min(max(split_fraction, 0.05), 0.95)
    split_idx = round(len(work_records) * split_fraction)
    split_idx = max(1, min(split_idx, len(work_records) - 1))

    first = work_records[:split_idx]
    second = work_records[split_idx:]

    hr1 = _safe_avg([r.get("heart_rate_bpm") for r in first])
    hr2 = _safe_avg([r.get("heart_rate_bpm") for r in second])

    def _has_power(recs):
        return any(r.get("power_w") is not None for r in recs)

    # Power source priority: developer-field power -> native power -> pace
    # fallback. power_w itself is already source-merged per record (native
    # preferred when a single record somehow has both); this just labels
    # which source is actually feeding this window.
    sources = {r.get("power_source") for r in (first + second) if r.get("power_source")}
    if "developer_field" in sources:
        power_method = "developer_field_power"
    elif "native" in sources:
        power_method = "native_power"
    else:
        power_method = None

    result: Dict[str, Any] = {}

    if power_method and _has_power(first) and _has_power(second) and hr1 and hr2:
        method = power_method
        metric1 = _compute_normalized_power([r.get("power_w") for r in first])
        metric2 = _compute_normalized_power([r.get("power_w") for r in second])
        ratio1 = (metric1 / hr1) if metric1 and hr1 else None
        ratio2 = (metric2 / hr2) if metric2 and hr2 else None
    else:
        method = "pace_fallback"
        metric1 = _safe_avg([r.get("speed_mps") for r in first])
        metric2 = _safe_avg([r.get("speed_mps") for r in second])
        ratio1 = (metric1 / hr1) if metric1 and hr1 else None
        ratio2 = (metric2 / hr2) if metric2 and hr2 else None

    if ratio1 is None or ratio2 is None or ratio1 == 0:
        return {
            "error": "insufficient_data",
            "message": "Not enough HR + power/pace data in both halves to compute decoupling.",
        }

    decoupling_pct = ((ratio2 - ratio1) / ratio1) * 100
    if abs(decoupling_pct) < 5:
        interpretation = "well_coupled"
    elif abs(decoupling_pct) < 10:
        interpretation = "moderate_drift"
    else:
        interpretation = "significant_decoupling"

    result["method"] = method
    result["split_fraction"] = round(split_fraction, 2)
    result["decoupling_pct"] = round(decoupling_pct, 1)
    result["interpretation"] = interpretation
    result["note"] = "Negative = HR rose relative to power/pace (decoupling). >10% suggests aerobic base is limiting."

    if method == "pace_fallback":
        result["first_half"] = {"duration_s": len(first), "avg_hr_bpm": round(hr1, 1), "avg_speed_mps": round(metric1, 3)}
        result["second_half"] = {"duration_s": len(second), "avg_hr_bpm": round(hr2, 1), "avg_speed_mps": round(metric2, 3)}
    else:
        result["first_half"] = {"duration_s": len(first), "avg_hr_bpm": round(hr1, 1), "normalized_power_w": round(metric1)}
        result["second_half"] = {"duration_s": len(second), "avg_hr_bpm": round(hr2, 1), "normalized_power_w": round(metric2)}

    # Cadence fade
    cad1 = _safe_avg([r.get("cadence_rpm") for r in first])
    cad2 = _safe_avg([r.get("cadence_rpm") for r in second])
    if cad1 is not None and cad2 is not None:
        result[f"avg_cadence_first_half_{cadence_unit}"] = round(cad1 * cadence_mult)
        result[f"avg_cadence_second_half_{cadence_unit}"] = round(cad2 * cadence_mult)

    # Grade guard: net elevation change (end - start altitude) per half.
    alt1 = [r["altitude_m"] for r in first if r.get("altitude_m") is not None]
    alt2 = [r["altitude_m"] for r in second if r.get("altitude_m") is not None]
    if len(alt1) >= 2 and len(alt2) >= 2:
        net1 = alt1[-1] - alt1[0]
        net2 = alt2[-1] - alt2[0]
        result["net_elevation_first_half_m"] = round(net1, 1)
        result["net_elevation_second_half_m"] = round(net2, 1)
        grade_confounded = abs(net1 - net2) > 30
        result["grade_confounded"] = grade_confounded
        if grade_confounded:
            result["grade_guard_note"] = (
                f"Net elevation differs by {round(abs(net1 - net2), 1)} m between halves — "
                "decoupling may reflect terrain, not fatigue."
            )

    return result


# ---------------------------------------------------------------------------
# Temperature correlation
# ---------------------------------------------------------------------------

def _compute_temperature_stats(records: List[Dict]) -> Optional[Dict]:
    """Summarize temperature and its relationship to HR and power."""
    temps = [r["temperature_c"] for r in records if r.get("temperature_c") is not None]
    if not temps:
        return None

    avg_temp = _safe_avg(temps)
    temp_min = min(temps)
    temp_max = max(temps)

    # Compare HR and power in coolest vs hottest third of records
    sorted_by_temp = sorted(
        [r for r in records if r.get("temperature_c") is not None],
        key=lambda r: r["temperature_c"]
    )
    third = max(1, len(sorted_by_temp) // 3)
    coolest = sorted_by_temp[:third]
    hottest = sorted_by_temp[-third:]

    cool_hr = _safe_round(_safe_avg([r["heart_rate_bpm"] for r in coolest if r.get("heart_rate_bpm")]))
    hot_hr = _safe_round(_safe_avg([r["heart_rate_bpm"] for r in hottest if r.get("heart_rate_bpm")]))
    cool_power = _safe_round(_safe_avg([r["power_w"] for r in coolest if r.get("power_w")]))
    hot_power = _safe_round(_safe_avg([r["power_w"] for r in hottest if r.get("power_w")]))

    result: Dict[str, Any] = {
        "avg_temp_c": _safe_round(avg_temp),
        "min_temp_c": temp_min,
        "max_temp_c": temp_max,
        "temp_range_c": round(temp_max - temp_min, 1),
    }
    if cool_hr is not None:
        result["avg_hr_coolest_third_bpm"] = cool_hr
    if hot_hr is not None:
        result["avg_hr_hottest_third_bpm"] = hot_hr
    if cool_power is not None:
        result["avg_power_coolest_third_w"] = cool_power
    if hot_power is not None:
        result["avg_power_hottest_third_w"] = hot_power

    return result


# ---------------------------------------------------------------------------
# Weight / W/kg lookup
# ---------------------------------------------------------------------------

def _get_rider_weight_kg(activity_date_str: str) -> Optional[float]:
    """Fetch rider weight from Garmin body composition for the given date."""
    if garmin_client is None:
        return None
    try:
        data = garmin_client.get_body_composition(activity_date_str)
        if not data:
            return None
        # Response is typically {"startDate": ..., "endDate": ..., "dateWeightList": [...]}
        weight_list = data.get("dateWeightList") or data.get("totalAverage") and [data]
        if not weight_list:
            # Try flat dict
            weight_g = data.get("weight") or data.get("totalWeightInGrams")
            if weight_g:
                return round(weight_g / 1000.0, 2)
            return None
        # Find closest entry to the activity date
        for entry in reversed(weight_list):
            weight_g = entry.get("weight") or entry.get("totalWeightInGrams")
            if weight_g:
                return round(weight_g / 1000.0, 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Shift summary
# ---------------------------------------------------------------------------

def _compute_shift_summary(shifts: list) -> dict:
    """Compute aggregate statistics over all shift events."""
    if not shifts:
        return {"total_shifts": 0, "note": "No DI2 shift events found in FIT file"}

    quality_counts: Dict[str, int] = {
        "proactive": 0, "reactive": 0, "coasting": 0, "spun_out": 0, "unknown": 0
    }
    terrain_quality: Dict[str, Dict[str, int]] = {
        "climbing": {"proactive": 0, "reactive": 0, "coasting": 0, "spun_out": 0, "unknown": 0},
        "flat": {"proactive": 0, "reactive": 0, "coasting": 0, "spun_out": 0, "unknown": 0},
        "descending": {"proactive": 0, "reactive": 0, "coasting": 0, "spun_out": 0, "unknown": 0},
    }
    front_shifts = 0
    rear_shifts = 0
    gear_usage: dict = {}
    cadences_at_shift = []

    # Burst detection: 3+ shifts within ~6 consecutive events
    panic_bursts = 0
    burst_i = 0
    while burst_i < len(shifts):
        window = [shifts[burst_i]]
        for j in range(burst_i + 1, min(burst_i + 6, len(shifts))):
            window.append(shifts[j])
        if len(window) >= 3:
            panic_bursts += 1
            burst_i += len(window)
        else:
            burst_i += 1

    for s in shifts:
        quality = s.get("quality", "unknown")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # Terrain classification
        grade = s.get("grade_at_shift_pct")
        if grade is not None:
            if grade >= 3:
                terrain = "climbing"
            elif grade <= -3:
                terrain = "descending"
            else:
                terrain = "flat"
        else:
            terrain = "flat"  # default
        terrain_quality[terrain][quality] = terrain_quality[terrain].get(quality, 0) + 1

        event = s.get("event", "")
        if "front" in event:
            front_shifts += 1
        else:
            rear_shifts += 1

        combo = s.get("gear_combo")
        if combo:
            gear_usage[combo] = gear_usage.get(combo, 0) + 1

        cad = s.get("cadence_at_shift_rpm")
        if cad is not None:
            cadences_at_shift.append(cad)

    total = len(shifts)
    proactive = quality_counts.get("proactive", 0)
    reactive = quality_counts.get("reactive", 0)

    summary: Dict[str, Any] = {
        "total_shifts": total,
        "front_shifts": front_shifts,
        "rear_shifts": rear_shifts,
        "proactive_shifts": proactive,
        "reactive_shifts": reactive,
        "coasting_shifts": quality_counts.get("coasting", 0),
        "spun_out_shifts": quality_counts.get("spun_out", 0),
        "proactive_pct": round(proactive / total * 100, 1) if total else 0,
        "reactive_pct": round(reactive / total * 100, 1) if total else 0,
        "panic_burst_episodes": panic_bursts,
        "gear_usage": dict(sorted(gear_usage.items(), key=lambda x: -x[1])),
    }

    # Include terrain breakdown only if grade data was present
    has_terrain_data = any(s.get("grade_at_shift_pct") is not None for s in shifts)
    if has_terrain_data:
        terrain_summary = {}
        for terrain, counts in terrain_quality.items():
            t_total = sum(counts.values())
            if t_total > 0:
                terrain_summary[terrain] = {
                    "total": t_total,
                    "proactive": counts.get("proactive", 0),
                    "reactive": counts.get("reactive", 0),
                    "reactive_pct": round(counts.get("reactive", 0) / t_total * 100, 1),
                }
        if terrain_summary:
            summary["by_terrain"] = terrain_summary

    if cadences_at_shift:
        summary["avg_cadence_at_shift_rpm"] = round(
            sum(cadences_at_shift) / len(cadences_at_shift), 1
        )
        summary["min_cadence_at_shift_rpm"] = min(cadences_at_shift)
        summary["max_cadence_at_shift_rpm"] = max(cadences_at_shift)

    return summary


# ---------------------------------------------------------------------------
# Main FIT parsing logic
# ---------------------------------------------------------------------------

def _compute_hrv_metrics(rr_intervals_s: List[float]) -> Optional[Dict]:
    """Compute standard time-domain HRV metrics from R-R intervals (seconds).

    Returns RMSSD, SDNN, pNN50, mean R-R, and count. These are the standard
    intra-workout HRV metrics used in sports science and HRV-guided training
    platforms (HRV4Training, Elite HRV, Polar's recovery metrics, etc.).

    Requires at least 10 R-R intervals to produce stable results.
    """
    if len(rr_intervals_s) < 10:
        return None

    rr_ms = [r * 1000.0 for r in rr_intervals_s]
    diffs = [rr_ms[i + 1] - rr_ms[i] for i in range(len(rr_ms) - 1)]

    # RMSSD: root mean square of successive differences
    squared = [d * d for d in diffs]
    rmssd = (sum(squared) / len(squared)) ** 0.5 if squared else 0.0

    # SDNN: standard deviation of all N-N intervals
    mean_rr = sum(rr_ms) / len(rr_ms)
    sdnn = (sum((r - mean_rr) ** 2 for r in rr_ms) / (len(rr_ms) - 1)) ** 0.5 if len(rr_ms) > 1 else 0.0

    # pNN50: percentage of pairs differing by more than 50 ms
    nn50 = sum(1 for d in diffs if abs(d) > 50.0)
    pnn50 = 100.0 * nn50 / len(diffs) if diffs else 0.0

    return {
        "rmssd_ms": round(rmssd, 1),
        "sdnn_ms": round(sdnn, 1),
        "pnn50_pct": round(pnn50, 2),
        "mean_rr_ms": round(mean_rr, 1),
        "mean_hr_bpm": round(60000.0 / mean_rr, 1) if mean_rr > 0 else None,
        "rr_count": len(rr_ms),
    }


def _parse_fit(fit_bytes: bytes, include_records: bool, min_duration_minutes: float = 60) -> dict:
    """Parse a FIT file and extract structured cycling/running data.

    Args:
        min_duration_minutes: Minimum duration required before HR drift is
            computed (see _compute_hr_drift).
    """
    fit_bytes = _extract_fit_bytes(fit_bytes)
    fitfile = fitparse.FitFile(io.BytesIO(fit_bytes))

    session: Dict[str, Any] = {}
    laps: List[Dict] = []
    shifts: List[Dict] = []
    records: List[Dict] = []
    # R-R intervals from the FIT 'hrv' message type, paired with the timestamp
    # of the most recent record message so we can bucket them per lap later.
    # Requires "Log HRV" enabled on the watch AND a chest strap paired.
    rr_pairs: List[tuple] = []  # (record_timestamp, rr_seconds)
    last_record_ts = None

    # Track last values for context at shift time
    last_cadence: Optional[float] = None
    last_grade: Optional[float] = None

    # ConnectIQ developer fields discovered via 'field_description' messages,
    # e.g. {"power": "Power", "form_power": "Form Power"}. Populated before
    # any 'record' message that uses them, since FIT files declare developer
    # fields ahead of the data that references them.
    dev_field_name_by_key: Dict[str, str] = {}
    developer_fields_found: set = set()

    for message in fitfile.get_messages():
        msg_type = message.name

        # ------------------------------------------------------------------
        # Session summary
        # ------------------------------------------------------------------
        if msg_type == "session":
            session = {
                "sport": _get_field(message, "sport"),
                "sub_sport": _get_field(message, "sub_sport"),
                "start_time": str(_get_field(message, "start_time") or ""),
                "total_elapsed_time_s": _get_field(message, "total_elapsed_time"),
                "total_timer_time_s": _get_field(message, "total_timer_time"),
                "total_distance_m": _get_field(message, "total_distance"),
                "total_calories": _get_field(message, "total_calories"),
                "avg_speed_mps": _get_field(message, "avg_speed"),
                "max_speed_mps": _get_field(message, "max_speed"),
                "avg_power_w": _get_field(message, "avg_power"),
                "max_power_w": _get_field(message, "max_power"),
                "normalized_power_w": _get_field(message, "normalized_power"),
                "avg_cadence_rpm": _get_field(message, "avg_cadence"),
                "max_cadence_rpm": _get_field(message, "max_cadence"),
                "avg_heart_rate_bpm": _get_field(message, "avg_heart_rate"),
                "max_heart_rate_bpm": _get_field(message, "max_heart_rate"),
                "total_ascent_m": _get_field(message, "total_ascent"),
                "total_descent_m": _get_field(message, "total_descent"),
                "total_training_effect": _get_field(message, "total_training_effect"),
                "avg_left_power_phase_start_deg": _get_field(message, "avg_left_power_phase"),
                "avg_right_power_phase_start_deg": _get_field(message, "avg_right_power_phase"),
                "avg_left_pco_mm": _get_field(message, "avg_left_pco"),
                "avg_right_pco_mm": _get_field(message, "avg_right_pco"),
                "avg_left_torque_effectiveness_pct": _get_field(message, "avg_left_torque_effectiveness"),
                "avg_right_torque_effectiveness_pct": _get_field(message, "avg_right_torque_effectiveness"),
                "avg_left_pedal_smoothness_pct": _get_field(message, "avg_left_pedal_smoothness"),
                "avg_right_pedal_smoothness_pct": _get_field(message, "avg_right_pedal_smoothness"),
            }
            balance_raw = _get_field(message, "avg_left_right_balance")
            session["avg_left_power_pct"] = _decode_left_right_balance(balance_raw)
            if session["avg_left_power_pct"] is not None:
                session["avg_right_power_pct"] = round(100.0 - session["avg_left_power_pct"], 1)
            session = {k: v for k, v in session.items() if v is not None}

        # ------------------------------------------------------------------
        # Developer field discovery (Stryd etc.) — must run before any
        # 'record' message that references these fields; see _STRYD_ALIAS_MAP.
        # ------------------------------------------------------------------
        elif msg_type == "field_description":
            field_name_raw = _get_field(message, "field_name")
            if field_name_raw:
                name_str = str(field_name_raw).strip()
                if name_str:
                    developer_fields_found.add(name_str)
                    key = _STRYD_ALIAS_MAP.get(name_str.lower())
                    if key:
                        dev_field_name_by_key[key] = name_str

        # ------------------------------------------------------------------
        # Lap data
        # ------------------------------------------------------------------
        elif msg_type == "lap":
            lap: Dict[str, Any] = {
                "lap_number": len(laps) + 1,
                "start_time": str(_get_field(message, "start_time") or ""),
                "total_elapsed_time_s": _get_field(message, "total_elapsed_time"),
                "total_distance_m": _get_field(message, "total_distance"),
                "avg_speed_mps": _get_field(message, "avg_speed"),
                "avg_power_w": _get_field(message, "avg_power"),
                "normalized_power_w": _get_field(message, "normalized_power"),
                "avg_cadence_rpm": _get_field(message, "avg_cadence"),
                "avg_heart_rate_bpm": _get_field(message, "avg_heart_rate"),
                "avg_left_pco_mm": _get_field(message, "avg_left_pco"),
                "avg_right_pco_mm": _get_field(message, "avg_right_pco"),
                "avg_left_torque_effectiveness_pct": _get_field(message, "avg_left_torque_effectiveness"),
                "avg_right_torque_effectiveness_pct": _get_field(message, "avg_right_torque_effectiveness"),
                "avg_left_pedal_smoothness_pct": _get_field(message, "avg_left_pedal_smoothness"),
                "avg_right_pedal_smoothness_pct": _get_field(message, "avg_right_pedal_smoothness"),
                "total_ascent_m": _get_field(message, "total_ascent"),
                "total_descent_m": _get_field(message, "total_descent"),
                "intensity": _get_field(message, "intensity"),
            }
            balance_raw = _get_field(message, "avg_left_right_balance")
            left_pct = _decode_left_right_balance(balance_raw)
            if left_pct is not None:
                lap["avg_left_power_pct"] = left_pct
                lap["avg_right_power_pct"] = round(100.0 - left_pct, 1)
            lap = {k: v for k, v in lap.items() if v is not None}

            # Variability Index per lap
            np_w = lap.get("normalized_power_w")
            avg_w = lap.get("avg_power_w")
            if np_w and avg_w and avg_w > 0:
                lap["variability_index"] = round(np_w / avg_w, 3)

            laps.append(lap)

        # ------------------------------------------------------------------
        # DI2 / electronic shifting events
        # ------------------------------------------------------------------
        elif msg_type == "event":
            event_type = _get_field(message, "event")
            if event_type in ("rear_gear_change", "front_gear_change", "gear_change"):
                gear_data_raw = _get_field(message, "gear_change_data", "data")
                timestamp = _get_field(message, "timestamp")

                shift_entry: Dict[str, Any] = {
                    "timestamp": str(timestamp or ""),
                    "event": str(event_type),
                    "cadence_at_shift_rpm": last_cadence,
                }

                if last_grade is not None:
                    shift_entry["grade_at_shift_pct"] = round(last_grade, 1)

                if gear_data_raw is not None:
                    try:
                        decoded = _decode_gear_change(int(gear_data_raw))
                        shift_entry.update(decoded)
                        ft = decoded.get("front_teeth")
                        rt = decoded.get("rear_teeth")
                        if ft and rt:
                            shift_entry["gear_combo"] = f"{ft}/{rt}t"
                    except (TypeError, ValueError):
                        shift_entry["gear_change_data_raw"] = str(gear_data_raw)

                # Classify shift quality
                cad = last_cadence
                if cad is not None:
                    if cad == 0:
                        shift_entry["quality"] = "coasting"
                    elif cad < 70:
                        shift_entry["quality"] = "reactive"
                    elif cad > 100:
                        shift_entry["quality"] = "spun_out"
                    else:
                        shift_entry["quality"] = "proactive"
                else:
                    shift_entry["quality"] = "unknown"

                shift_entry = {k: v for k, v in shift_entry.items() if v is not None}
                shifts.append(shift_entry)

        # ------------------------------------------------------------------
        # Per-second records
        # ------------------------------------------------------------------
        elif msg_type == "record":
            cadence = _get_field(message, "cadence")
            if cadence is not None:
                last_cadence = cadence

            grade = _get_field(message, "grade")
            if grade is not None:
                last_grade = grade

            # Track timestamp for HRV bucketing
            ts = _get_field(message, "timestamp")
            if ts is not None:
                last_record_ts = ts

            # Power source priority: native power-meter field first, then a
            # Stryd/ConnectIQ developer field (running activities have no
            # native power meter). power_source records which one fed
            # power_w, for transparency and for get_activity_decoupling.
            native_power = _get_field(message, "power")
            dev_power = _dev_value(message, dev_field_name_by_key, "power")
            if native_power is not None:
                power_w = native_power
                power_source = "native"
            elif dev_power is not None:
                power_w = dev_power
                power_source = "developer_field"
            else:
                power_w = None
                power_source = None

            record: Dict[str, Any] = {
                "timestamp": str(_get_field(message, "timestamp") or ""),
                "power_w": power_w,
                "power_source": power_source,
                "cadence_rpm": cadence,
                "heart_rate_bpm": _get_field(message, "heart_rate"),
                "speed_mps": _get_field(message, "speed"),
                "altitude_m": _get_field(message, "altitude"),
                "grade_pct": grade,
                "temperature_c": _get_field(message, "temperature"),
                "lat_deg": _semicircles_to_degrees(_get_field(message, "position_lat")),
                "lon_deg": _semicircles_to_degrees(_get_field(message, "position_long")),
                "left_pco_mm": _get_field(message, "left_pco"),
                "right_pco_mm": _get_field(message, "right_pco"),
                "left_torque_effectiveness_pct": _get_field(message, "left_torque_effectiveness"),
                "right_torque_effectiveness_pct": _get_field(message, "right_torque_effectiveness"),
                "left_pedal_smoothness_pct": _get_field(message, "left_pedal_smoothness"),
                "right_pedal_smoothness_pct": _get_field(message, "right_pedal_smoothness"),
                "form_power_w": _dev_value(message, dev_field_name_by_key, "form_power"),
                "leg_spring_stiffness_kn_m": _dev_value(message, dev_field_name_by_key, "leg_spring_stiffness"),
                "air_power_w": _dev_value(message, dev_field_name_by_key, "air_power"),
                "vertical_oscillation_mm": _dev_value(message, dev_field_name_by_key, "vertical_oscillation"),
                "ground_contact_time_ms": _dev_value(message, dev_field_name_by_key, "ground_time"),
            }
            balance_raw = _get_field(message, "left_right_balance")
            left_pct = _decode_left_right_balance(balance_raw)
            if left_pct is not None:
                record["left_power_pct"] = left_pct
                record["right_power_pct"] = round(100.0 - left_pct, 1)

            left_phase = _get_field(message, "left_power_phase")
            if left_phase:
                try:
                    record["left_power_phase_start_deg"] = left_phase[0]
                    record["left_power_phase_end_deg"] = left_phase[1]
                except (IndexError, TypeError):
                    pass
            right_phase = _get_field(message, "right_power_phase")
            if right_phase:
                try:
                    record["right_power_phase_start_deg"] = right_phase[0]
                    record["right_power_phase_end_deg"] = right_phase[1]
                except (IndexError, TypeError):
                    pass

            record = {k: v for k, v in record.items() if v is not None}
            records.append(record)

        # ------------------------------------------------------------------
        # HRV — R-R intervals (one or more per message in field 'time')
        # ------------------------------------------------------------------
        elif msg_type == "hrv":
            rr_field = _get_field(message, "time")
            if rr_field is None:
                continue
            if not isinstance(rr_field, (list, tuple)):
                rr_field = [rr_field]
            for rr in rr_field:
                # Filter sentinel/invalid values. FIT spec uses ~65.535 s
                # as "no R-R interval detected" filler in fixed-size arrays.
                if rr is not None and 0.2 < rr < 3.0:
                    rr_pairs.append((last_record_ts, float(rr)))

    # ------------------------------------------------------------------
    # Post-parse enrichment
    # ------------------------------------------------------------------

    # Variability Index for session
    np_w = session.get("normalized_power_w")
    avg_w = session.get("avg_power_w")
    if np_w and avg_w and avg_w > 0:
        session["variability_index"] = round(np_w / avg_w, 3)

    # Backfill session/lap power aggregates from per-second records when the
    # device didn't precompute them natively — i.e. running activities where
    # power comes from a Stryd developer field rather than a power meter.
    # Native (device-computed) aggregates are left untouched.
    if records:
        if session.get("avg_power_w") is None:
            for k, v in _compute_power_aggregates_for_range(records).items():
                session.setdefault(k, v)

        for lap in laps:
            if lap.get("avg_power_w") is not None:
                continue
            lap_start = _parse_iso(lap.get("start_time"))
            elapsed = lap.get("total_elapsed_time_s")
            if lap_start is None or not elapsed:
                continue
            lap_end = lap_start + _dt.timedelta(seconds=float(elapsed))
            for k, v in _compute_power_aggregates_for_range(records, start_ts=lap_start, end_ts=lap_end).items():
                lap[k] = v

    # Cadence unit fix: FIT running cadence is per-leg rpm (e.g. 79 rpm =
    # 158 spm). Only running activities record cadence this way — cycling's
    # avg_cadence_rpm is genuine crank RPM and must not be doubled.
    is_running = str(session.get("sport", "")).lower() == "running"
    if is_running:
        if session.get("avg_cadence_rpm") is not None:
            session["avg_cadence_spm"] = round(session["avg_cadence_rpm"] * 2)
        for lap in laps:
            if lap.get("avg_cadence_rpm") is not None:
                lap["avg_cadence_spm"] = round(lap["avg_cadence_rpm"] * 2)

    shift_summary = _compute_shift_summary(shifts)

    # Record-based analytics (computed from per-second data regardless of include_records flag)
    grade_stats = _grade_analysis(records) if records else None
    hr_drift = _compute_hr_drift(records, min_duration_minutes=min_duration_minutes) if records else None
    temp_stats = _compute_temperature_stats(records) if records else None
    climbs = _detect_climbs(records) if records else []
    pdc = _compute_power_duration_curve(records) if records else None

    if grade_stats:
        session["grade_analysis"] = grade_stats
    if hr_drift:
        session["hr_drift"] = hr_drift
    if temp_stats:
        session["temperature_stats"] = temp_stats

    result: Dict[str, Any] = {
        "session": session,
        "laps": laps,
        "shift_summary": shift_summary,
        "shifts": shifts,
    }

    if developer_fields_found:
        result["developer_fields_found"] = sorted(developer_fields_found)

    if climbs:
        result["climbs"] = climbs

    if pdc:
        result["power_duration_curve"] = pdc

    # HRV (time-domain) — always include summary if R-R data exists.
    # Raw R-R array only included when include_records=True (can be large).
    # Also compute per-lap HRV by bucketing R-R intervals by timestamp.
    if rr_pairs:
        all_rr = [rr for (_, rr) in rr_pairs]
        hrv_summary = _compute_hrv_metrics(all_rr)
        if hrv_summary:
            result["hrv"] = hrv_summary

        # Per-lap HRV: walk laps in order, derive each lap's [start, end)
        # window from start_time + total_elapsed_time_s, filter R-R pairs.
        for lap in laps:
            lap_start = _parse_iso(lap.get("start_time"))
            elapsed = lap.get("total_elapsed_time_s")
            if lap_start is None or not elapsed:
                continue
            lap_end = lap_start + _dt.timedelta(seconds=float(elapsed))

            # Filter R-R pairs whose record_ts falls within this lap.
            # rr_pairs[i][0] is a datetime-like value from fitparse.
            lap_rr = []
            for ts, rr in rr_pairs:
                ts_dt = ts if isinstance(ts, _dt.datetime) else _parse_iso(ts)
                if ts_dt is None:
                    continue
                # Compare naively if either is tz-naive (FIT timestamps are UTC)
                if lap_start.tzinfo and not ts_dt.tzinfo:
                    ts_dt = ts_dt.replace(tzinfo=lap_start.tzinfo)
                elif ts_dt.tzinfo and not lap_start.tzinfo:
                    lap_start_cmp = lap_start.replace(tzinfo=ts_dt.tzinfo)
                    lap_end_cmp = lap_end.replace(tzinfo=ts_dt.tzinfo)
                    if lap_start_cmp <= ts_dt < lap_end_cmp:
                        lap_rr.append(rr)
                    continue
                if lap_start <= ts_dt < lap_end:
                    lap_rr.append(rr)

            lap_hrv = _compute_hrv_metrics(lap_rr)
            if lap_hrv:
                lap["hrv"] = lap_hrv

        if include_records:
            # Raw stream — list of {timestamp, rr_seconds} for full transparency
            result["rr_intervals_seconds"] = [
                {"timestamp": str(ts) if ts else None, "rr_seconds": rr}
                for (ts, rr) in rr_pairs
            ]

    if include_records:
        result["records"] = records

    return result


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

def register_tools(app):
    """Register all activity analysis tools with the MCP server app"""

    @app.tool()
    async def get_activity_fit_data(
        activity_id: Union[int, str],
        include_records: bool = False,
        min_duration_minutes: float = 60,
    ) -> str:
        """Download and parse FIT file for an activity to expose advanced cycling/running data.

        Returns data not available through the standard REST API, including:
        - DI2 / electronic shifting events with cadence at time of shift, grade at shift,
          gear combinations, shift quality classification, and terrain-grouped shift analysis
        - Cycling dynamics per session and lap: platform center offset (PCO), left/right power
          balance, torque effectiveness, pedal smoothness
        - Running power from Stryd/ConnectIQ developer fields when no native power meter is
          present: avg_power_w, max_power_w, normalized_power_w (30s rolling, 4th-power mean),
          variability_index, ef (NP ÷ avg HR), avg_form_power_w, avg_leg_spring_stiffness_kn_m
          per session and lap, plus a developer_fields_found list of whatever ConnectIQ fields
          were discovered (so unrecognized ones stay visible). avg_cadence_spm is added for
          running activities (native cadence is per-leg rpm, i.e. half of total steps/min).
        - Variability Index (NP / avg_power) per session and lap
        - Climb detection with VAM (vertical ascent rate), avg power/cadence/HR per climb,
          and W/kg per climb (using auto-fetched body weight from Garmin)
        - Grade-correlated stats: avg power, cadence, HR broken down by terrain steepness
        - HR drift / cardiac drift coefficient (aerobic decoupling, gated by min_duration_minutes)
        - Temperature correlation: avg HR/power in hottest vs. coolest portions of ride
        - Power Duration Curve: best mean maximal power at 5s, 30s, 1min, 5min, 10min, 20min, 60min
        - Optional full per-second time series when include_records=True

        Shift quality:
        - proactive: shifted at 70-100 rpm (ideal cadence range)
        - reactive: shifted below 70 rpm (already grinding before shifting)
        - coasting: shifted at 0 rpm (mid-stop or freewheeling)
        - spun_out: shifted above 100 rpm (waited too long in easy gear)

        Note: DI2 data requires Shimano Di2 / SRAM eTap. Cycling dynamics require a
        compatible power meter (e.g., Garmin Rally, Favero Assioma, PowerTap P1 pedals).

        Args:
            activity_id: Garmin activity ID
            include_records: Include full per-second time series (default False).
                             Warning: adds significant data volume for long rides.
            min_duration_minutes: Minimum activity duration required before HR drift
                             is computed (default 60).
        """
        if not FITPARSE_AVAILABLE:
            return (
                "fitparse library is not installed. "
                "Install it with: pip install fitparse"
            )

        try:
            activity_id = int(activity_id)
            from garminconnect import Garmin

            fit_bytes = garmin_client.download_activity(
                activity_id,
                dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
            )

            if not fit_bytes:
                return f"No FIT data returned for activity {activity_id}"

            raw = bytes(fit_bytes)

            try:
                parsed = _parse_fit(raw, include_records=include_records, min_duration_minutes=min_duration_minutes)
            except Exception as parse_err:
                return json.dumps({
                    "error": str(parse_err),
                    "debug": {
                        "total_bytes": len(raw),
                        "first_16_bytes_hex": raw[:16].hex(),
                        "hint": (
                            "1f8b = gzip, 504b = ZIP, 0e10/0c10 = raw FIT, "
                            "3c or 7b = HTML/JSON error from Garmin"
                        ),
                    }
                }, indent=2)

            parsed["activity_id"] = activity_id

            # W/kg: fetch body weight matched to activity date
            start_time_str = parsed.get("session", {}).get("start_time", "")
            activity_date = start_time_str[:10] if start_time_str else None
            if activity_date:
                weight_kg = _get_rider_weight_kg(activity_date)
                if weight_kg:
                    parsed["rider_weight_kg"] = weight_kg
                    # W/kg for session avg power and NP
                    avg_w = parsed["session"].get("avg_power_w")
                    np_w = parsed["session"].get("normalized_power_w")
                    if avg_w:
                        parsed["session"]["avg_w_per_kg"] = round(avg_w / weight_kg, 2)
                    if np_w:
                        parsed["session"]["normalized_w_per_kg"] = round(np_w / weight_kg, 2)
                    # W/kg per climb
                    for climb in parsed.get("climbs", []):
                        climb_w = climb.get("avg_power_w")
                        if climb_w:
                            climb["avg_w_per_kg"] = round(climb_w / weight_kg, 2)

            return json.dumps(parsed, indent=2, default=str)

        except Exception as e:
            return f"Error downloading FIT data for activity {activity_id}: {str(e)}"

    @app.tool()
    async def get_activity_decoupling(
        activity_id: Union[int, str],
        split: Optional[float] = None,
        min_duration_minutes: float = 45,
    ) -> str:
        """Compute Pw:Hr (or Pa:Hr) aerobic decoupling between the first and second
        half of an activity's *work* portion — a compact (<1KB) coaching signal,
        never the underlying time series.

        - Excludes warmup/cooldown laps (via the FIT lap 'intensity' field) from
          the work portion when present; otherwise uses the whole activity.
        - Power source priority: developer-field power (e.g. Stryd) -> native
          power meter -> pace-based Pa:Hr fallback when no power data exists.
          The 'method' field reports which was used.
        - decoupling_pct: % change in (normalized_power_w or avg_speed_mps) ÷
          avg_hr_bpm from the first half to the second. Negative = HR rose
          relative to output (classic aerobic decoupling).
        - grade_confounded: true when the two halves' net elevation change
          differs by more than ~30 m — decoupling in that case may reflect
          terrain, not fatigue, so treat the number with caution.
        - Cadence fade: avg cadence (spm for running, rpm for cycling) per half.

        Args:
            activity_id: Garmin activity ID
            split: Fraction (0-1) of the work portion's moving time at which to
                   divide first/second half. Defaults to 0.5 (true halfway).
            min_duration_minutes: Minimum work-portion duration required before
                   decoupling is computed (default 45).
        """
        if not FITPARSE_AVAILABLE:
            return (
                "fitparse library is not installed. "
                "Install it with: pip install fitparse"
            )

        try:
            activity_id = int(activity_id)
            from garminconnect import Garmin

            fit_bytes = garmin_client.download_activity(
                activity_id,
                dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
            )

            if not fit_bytes:
                return f"No FIT data returned for activity {activity_id}"

            raw = bytes(fit_bytes)

            try:
                parsed = _parse_fit(raw, include_records=True)
            except Exception as parse_err:
                return json.dumps({
                    "error": str(parse_err),
                    "debug": {
                        "total_bytes": len(raw),
                        "first_16_bytes_hex": raw[:16].hex(),
                        "hint": (
                            "1f8b = gzip, 504b = ZIP, 0e10/0c10 = raw FIT, "
                            "3c or 7b = HTML/JSON error from Garmin"
                        ),
                    }
                }, indent=2)

            result = _compute_decoupling(parsed, split=split, min_duration_minutes=min_duration_minutes)
            result["activity_id"] = activity_id

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return f"Error computing decoupling for activity {activity_id}: {str(e)}"

    @app.tool()
    async def get_power_duration_curve(
        num_activities: int = 20,
        activity_type: str = "cycling",
    ) -> str:
        """Get season-best Power Duration Curve across recent activities.

        Downloads FIT files for recent cycling activities and computes best mean maximal
        power at each standard duration. Returns season bests with which activity and date
        each best came from.

        Durations: 5s (sprint), 30s, 1min, 5min (VO2 max proxy), 10min, 20min (FTP proxy), 60min

        Use the 20-minute best × 0.95 as a strong FTP estimate without a formal test.

        Warning: downloads multiple FIT files — may take 30-60 seconds for 20 activities.

        Args:
            num_activities: Number of recent activities to analyze (default 20, max 50)
            activity_type: Activity type to filter (default "cycling")
        """
        if not FITPARSE_AVAILABLE:
            return "fitparse library is not installed. Install it with: pip install fitparse"

        MAX_ACTIVITIES = 50
        num_activities = min(num_activities, MAX_ACTIVITIES)

        try:
            from garminconnect import Garmin

            # Fetch recent activities
            activities = garmin_client.get_activities(0, num_activities)
            if not activities:
                return "No activities found."

            # Filter by sport type
            cycling_activities = [
                a for a in activities
                if activity_type.lower() in str(a.get("activityType", {}).get("typeKey", "")).lower()
                or activity_type.lower() in str(a.get("activityType", {}).get("parentTypeId", "")).lower()
                or "cycling" in str(a.get("activityType", {}).get("typeKey", "")).lower()
                or "biking" in str(a.get("activityType", {}).get("typeKey", "")).lower()
            ]

            if not cycling_activities:
                return f"No {activity_type} activities found in the last {num_activities} activities."

            # Season bests: duration_label -> {watts, activity_id, date}
            season_bests: Dict[str, Any] = {}
            processed = 0
            errors = 0

            for activity in cycling_activities:
                act_id = activity.get("activityId")
                act_date = str(activity.get("startTimeLocal", ""))[:10]
                if not act_id:
                    continue

                try:
                    fit_bytes = garmin_client.download_activity(
                        act_id,
                        dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
                    )
                    if not fit_bytes:
                        continue

                    raw = bytes(fit_bytes)
                    extracted = _extract_fit_bytes(raw)
                    fitfile = fitparse.FitFile(io.BytesIO(extracted))

                    # Extract just the power records
                    act_records = []
                    for msg in fitfile.get_messages("record"):
                        p = msg.get_value("power")
                        act_records.append({"power_w": p})

                    pdc = _compute_power_duration_curve(act_records)
                    if pdc:
                        for label, watts in pdc.items():
                            if label not in season_bests or watts > season_bests[label]["watts"]:
                                season_bests[label] = {
                                    "watts": watts,
                                    "activity_id": act_id,
                                    "date": act_date,
                                }
                    processed += 1

                except Exception:
                    errors += 1
                    continue

            if not season_bests:
                return f"No power data found across {processed} activities."

            # Order by duration
            ordered = {label: season_bests[label] for label in _PDC_LABELS if label in season_bests}

            # FTP estimate from 20min best
            ftp_estimate = None
            if "20min" in ordered:
                ftp_estimate = round(ordered["20min"]["watts"] * 0.95)

            return json.dumps({
                "activities_analyzed": processed,
                "activities_skipped": errors,
                "ftp_estimate_w": ftp_estimate,
                "ftp_note": "Estimated as 95% of 20-minute best power",
                "season_bests": ordered,
            }, indent=2)

        except Exception as e:
            return f"Error computing power duration curve: {str(e)}"

    @app.tool()
    async def download_activity_file(
        activity_id: Union[int, str],
        format: str = "fit",
        output_dir: Optional[str] = None,
        return_base64: bool = False,
    ) -> str:
        """Download an activity and save it to disk as a file.

        Saves the activity in the requested format. Defaults to the original .fit
        file; Garmin also supports gpx, tcx, and csv.

        Directory resolution (first match wins):
          1. output_dir argument (one-off; not persisted)
          2. GARMIN_FIT_DOWNLOAD_DIR environment variable
          3. persisted config (set via set_fit_download_dir)
        If none is configured AND return_base64 is False, returns status
        "needs_setup" with a suggested default (the server's current working
        directory). In that case, ask the user where to save, call
        set_fit_download_dir(path), then call this tool again.

        Files are named "{activity_id}.{ext}" and overwrite any existing file.

        Args:
            activity_id: Garmin activity ID
            format: One of fit, gpx, tcx, csv (default fit)
            output_dir: Optional one-off directory override (not persisted)
            return_base64: Also return the file content as a base64 string
                           (capped at 5 MB). Use this when the calling client
                           can't reach the MCP server's filesystem — no
                           download directory is required in that case.
        """
        try:
            fmt = str(format).strip().lower()
            from garminconnect import Garmin

            format_map = {
                "fit": Garmin.ActivityDownloadFormat.ORIGINAL,
                "gpx": Garmin.ActivityDownloadFormat.GPX,
                "tcx": Garmin.ActivityDownloadFormat.TCX,
                "csv": Garmin.ActivityDownloadFormat.CSV,
            }
            if fmt not in format_map:
                return json.dumps({
                    "error": f"Invalid format '{format}'.",
                    "valid_formats": list(format_map.keys()),
                }, indent=2)

            download_dir = _resolve_download_dir(output_dir)
            if download_dir is None and not return_base64:
                return json.dumps({
                    "status": "needs_setup",
                    "suggested_default": os.getcwd(),
                    "config_path": os.path.expanduser(_get_fit_config_path()),
                    "message": (
                        "No download directory configured. Ask the user where to "
                        "save activity files (offer the current working directory "
                        "as the default), then call set_fit_download_dir(path) "
                        "before downloading, or pass return_base64=true to skip "
                        "saving to disk."
                    ),
                }, indent=2)

            activity_id = int(activity_id)

            data = garmin_client.download_activity(
                activity_id, dl_fmt=format_map[fmt]
            )
            if not data:
                return f"No {fmt} data returned for activity {activity_id}"

            raw = bytes(data)
            if fmt == "fit":
                try:
                    payload = _extract_fit_bytes(raw)
                except Exception as extract_err:
                    return json.dumps({
                        "error": str(extract_err),
                        "debug": {
                            "total_bytes": len(raw),
                            "first_16_bytes_hex": raw[:16].hex(),
                            "hint": (
                                "1f8b = gzip, 504b = ZIP, 0e10/0c10 = raw FIT, "
                                "3c or 7b = HTML/JSON error from Garmin"
                            ),
                        },
                    }, indent=2)
            else:
                payload = raw

            response: Dict[str, Any] = {
                "activity_id": activity_id,
                "format": fmt,
                "size_bytes": len(payload),
            }

            if download_dir is not None:
                os.makedirs(download_dir, exist_ok=True)
                file_path = os.path.join(download_dir, f"{activity_id}.{fmt}")
                with open(file_path, "wb") as f:
                    f.write(payload)
                response["file_path"] = os.path.abspath(file_path)

            if return_base64:
                MAX_BASE64_SOURCE_BYTES = 5 * 1024 * 1024
                if len(payload) > MAX_BASE64_SOURCE_BYTES:
                    response["base64_error"] = (
                        f"File is {len(payload)} bytes, exceeds the 5 MB return_base64 cap. "
                        "Use output_dir instead if the server filesystem is reachable."
                    )
                else:
                    response["content_base64"] = base64.b64encode(payload).decode("ascii")

            response["message"] = (
                "Activity file saved."
                if download_dir is not None
                else "Activity file returned as base64 (not saved; no download_dir configured)."
            )

            return json.dumps(response, indent=2)

        except Exception as e:
            return f"Error downloading activity {activity_id}: {str(e)}"

    @app.tool()
    async def set_fit_download_dir(path: str) -> str:
        """Set and persist the default directory for downloaded activity files.

        Stores the absolute path in a small JSON config file
        (~/.garminconnect_fit_config.json, overridable via GARMIN_FIT_CONFIG) so
        download_activity_file can save files without asking again.

        Args:
            path: Directory where activity files (.fit/.gpx/.tcx/.csv) are saved.
                  Pass the current working directory to keep files where the
                  server runs.
        """
        try:
            abspath = os.path.abspath(os.path.expanduser(path))
            os.makedirs(abspath, exist_ok=True)
            _write_fit_config(abspath)
            return json.dumps({
                "download_dir": abspath,
                "config_path": os.path.expanduser(_get_fit_config_path()),
                "message": "Default FIT download directory configured.",
            }, indent=2)
        except Exception as e:
            return f"Error setting FIT download directory: {str(e)}"

    return app
