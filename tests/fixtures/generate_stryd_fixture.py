"""Regenerates tests/fixtures/stryd_run.fit.

Not run as part of the test suite — requires the `fit-tool` package
(dev-only, not a project dependency: `pip install fit-tool`). Run manually
if the fixture needs to change:

    python tests/fixtures/generate_stryd_fixture.py

Builds a synthetic 50-minute run (5 min warmup + 45 min work) with Stryd-style
ConnectIQ developer fields (Power, Form Power, Leg Spring Stiffness, Air
Power) — no native power meter, mirroring a Garmin Fenix + Stryd footpod
setup. The work portion climbs ~54m in the first half and descends ~54m in
the second (net diff > 30m, to exercise the decoupling tool's grade guard),
and HR drifts up in the second half at similar power (to exercise
decoupling/HR-drift detection).
"""
import datetime
import math

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.developer_field import DeveloperField
from fit_tool.base_type import BaseType
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.developer_data_id_message import DeveloperDataIdMessage
from fit_tool.profile.messages.field_description_message import FieldDescriptionMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.profile_type import FileType, Manufacturer, Event, EventType, Sport, Intensity


def build(out_path: str):
    start = datetime.datetime(2026, 7, 22, 12, 46, 22, tzinfo=datetime.timezone.utc)
    start_ms = round(start.timestamp() * 1000)

    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    fid = FileIdMessage()
    fid.type = FileType.ACTIVITY
    fid.manufacturer = Manufacturer.GARMIN.value
    fid.product = 0
    fid.time_created = start_ms
    fid.serial_number = 0xDEADBEEF
    builder.add(fid)

    ddid = DeveloperDataIdMessage()
    ddid.developer_id = bytes(range(16))
    ddid.application_id = bytes(range(16))
    ddid.manufacturer_id = 0
    ddid.developer_data_index = 0
    builder.add(ddid)

    dev_fields_meta = [
        (0, "Power", "watts"),
        (1, "Form Power", "watts"),
        (2, "Leg Spring Stiffness", "kN/m"),
        (3, "Air Power", "watts"),
    ]
    for field_def_num, name, units in dev_fields_meta:
        fd = FieldDescriptionMessage()
        fd.developer_data_index = 0
        fd.field_definition_number = field_def_num
        fd.fit_base_type_id = BaseType.FLOAT32.value
        fd.field_name = name
        fd.units = units
        builder.add(fd)

    def _dev_field(field_id, name, value):
        f = DeveloperField(field_id=field_id, developer_data_index=0, name=name,
                            base_type=BaseType.FLOAT32, growable=True)
        f.set_value(0, float(value))
        return f

    def dev_fields(power=None, form_power=None, lss=None, air_power=None):
        vals = []
        if power is not None:
            vals.append(_dev_field(0, "Power", power))
        if form_power is not None:
            vals.append(_dev_field(1, "Form Power", form_power))
        if lss is not None:
            vals.append(_dev_field(2, "Leg Spring Stiffness", lss))
        if air_power is not None:
            vals.append(_dev_field(3, "Air Power", air_power))
        return vals

    ts = start_ms
    records = []

    ev = EventMessage()
    ev.event = Event.TIMER
    ev.event_type = EventType.START
    ev.timestamp = ts
    builder.add(ev)

    # --- Warmup: 300s, power ~150W, HR ~120bpm, flat altitude ---
    WARMUP_S = 300
    altitude = 100.0
    for i in range(WARMUP_S):
        rec = RecordMessage()
        rec.timestamp = ts
        rec.heart_rate = 115 + int(5 * math.sin(i / 30))
        rec.cadence = 75
        rec.altitude = altitude
        rec.speed = 2.2
        rec.developer_fields = dev_fields(power=150 + 10 * math.sin(i / 10), form_power=25, lss=8.5, air_power=2)
        records.append(rec)
        ts += 1000

    lap1 = LapMessage()
    lap1.timestamp = ts
    lap1.start_time = start_ms
    lap1.total_elapsed_time = WARMUP_S
    lap1.total_timer_time = WARMUP_S
    lap1.total_distance = 660
    lap1.intensity = Intensity.WARMUP
    builder.add_all(records)
    builder.add(lap1)
    records = []

    # --- Active/work portion: 2700s. First half climbs, second half descends
    # (net elevation diff >> 30m -> should trip grade_confounded). HR drifts
    # up in the second half at similar power -> positive decoupling.
    ACTIVE_S = 2700
    active_start_ms = ts
    for i in range(ACTIVE_S):
        rec = RecordMessage()
        rec.timestamp = ts
        if i < ACTIVE_S // 2:
            altitude += 0.04  # climbing ~54m over first half
            hr_base = 145
        else:
            altitude -= 0.04  # descending ~54m over second half
            hr_base = 158  # HR drifted up for similar power -> decoupling
        rec.heart_rate = hr_base + int(4 * math.sin(i / 45))
        rec.cadence = 79
        rec.altitude = altitude
        rec.speed = 2.6
        power = 274 + 20 * math.sin(i / 8) + (10 if i % 37 == 0 else 0)
        rec.developer_fields = dev_fields(power=power, form_power=28, lss=9.0, air_power=3)
        records.append(rec)
        ts += 1000

    lap2 = LapMessage()
    lap2.timestamp = ts
    lap2.start_time = active_start_ms
    lap2.total_elapsed_time = ACTIVE_S
    lap2.total_timer_time = ACTIVE_S
    lap2.total_distance = 7100
    lap2.intensity = Intensity.ACTIVE
    builder.add_all(records)
    builder.add(lap2)

    ev = EventMessage()
    ev.event = Event.TIMER
    ev.event_type = EventType.STOP
    ev.timestamp = ts
    builder.add(ev)

    sess = SessionMessage()
    sess.timestamp = ts
    sess.start_time = start_ms
    sess.total_elapsed_time = WARMUP_S + ACTIVE_S
    sess.total_timer_time = WARMUP_S + ACTIVE_S
    sess.total_distance = 7760
    sess.sport = Sport.RUNNING
    sess.avg_heart_rate = 146
    sess.avg_cadence = 79
    builder.add(sess)

    fit_file = builder.build()
    fit_file.to_file(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import os
    build(os.path.join(os.path.dirname(__file__), "stryd_run.fit"))
