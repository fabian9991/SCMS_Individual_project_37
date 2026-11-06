"""Read-only controller diagnostics. Never supplies state or commands to control."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_configuration(path, configuration, sources=()):
    payload = dict(configuration)
    payload['source_sha256'] = {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in sources if Path(p).is_file()
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=json_value) + '\n',
                          encoding='utf-8')


class ControllerLog:
    """Separate CSV keeps the existing trajectory.csv columns unchanged.

    Wheel rates describe the preceding interval. Requests/servo outputs are
    commands for the following interval; both interval endpoints are logged.
    Route errors use the exact segment examined by Tracker.command, including
    calls which subsequently switch segments. Only lqr_active rows are LQR data.
    """
    FIELDS = [
        'time_s', 'rate_interval_start_s', 'control_dt_s', 'mission_goal_index',
        'route_id', 'route_segment_index', 'status', 'navigator_mode',
        'tracker_mode', 'planner_status', 'reference_valid', 'lqr_active',
        'ref_ax_m', 'ref_ay_m', 'ref_bx_m', 'ref_by_m', 'ref_heading_rad',
        'estimated_cross_track_m', 'actual_cross_track_m',
        'estimated_heading_error_rad', 'actual_heading_error_rad',
        'requested_v_m_s', 'raw_lqr_omega_rad_s', 'requested_omega_rad_s',
        'omega_limit_active', 'wheel_scale', 'scaled_v_m_s',
        'scaled_omega_rad_s', 'desired_left_rad_s', 'desired_right_rad_s',
        'measured_left_rad_s', 'measured_right_rad_s',
        'previous_desired_left_rad_s', 'previous_desired_right_rad_s',
        'previous_stop', 'stop', 'left_servo_raw_rad_s', 'right_servo_raw_rad_s',
        'left_command_rad_s', 'right_command_rad_s',
        'left_servo_limit_active', 'right_servo_limit_active',
        'left_integral_rad_s', 'right_integral_rad_s',
        'predicted_clearance_m', 'est_x_m', 'est_y_m', 'est_heading_rad',
        'true_x_m', 'true_y_m', 'true_heading_rad',
    ]

    def __init__(self, directory, wheel_radius, wheel_separation):
        self.directory = Path(directory)
        self.radius, self.separation = wheel_radius, wheel_separation
        self.handle = (self.directory / 'controller.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.event_handle = (self.directory / 'controller_events.csv').open('w', newline='', encoding='utf-8')
        self.events = csv.DictWriter(self.event_handle,
                                    fieldnames=['time_s', 'event', 'details_json'])
        self.events.writeheader()
        self.previous_desired = np.zeros(2)
        self.previous_stop = True
        self.next_flush = 0.0
        self.closed = False
        self.previous_status = None

    def event(self, t, event, **details):
        self.events.writerow(dict(time_s=t, event=event,
                                  details_json=json.dumps(details, default=json_value)))

    def record(self, t, elapsed, navigator, servo, pose, truth, v, omega,
               desired, measured, commands, stop, status, clearance):
        for event in navigator.events:
            self.event(event['time_s'], event['event'], **event['details'])
        navigator.events.clear()
        if status != self.previous_status:
            self.event(t, 'status_change', previous=self.previous_status, current=status,
                       route_id=navigator.route_id, goal_index=navigator.segment)
            self.previous_status = status
        d = navigator.tracker.diagnostics
        valid = bool(d) and navigator.mode != 'DONE'
        row = {k: np.nan for k in self.FIELDS}
        if valid:
            a, b = np.asarray(d['a']), np.asarray(d['b'])
            phi = float(np.arctan2(*(b - a)[::-1]))
            normal = np.array([-np.sin(phi), np.cos(phi)])
            row.update(route_segment_index=d['segment'], ref_ax_m=a[0], ref_ay_m=a[1],
                       ref_bx_m=b[0], ref_by_m=b[1], ref_heading_rad=phi,
                       estimated_cross_track_m=float((pose[:2] - a) @ normal),
                       actual_cross_track_m=float((truth[:2] - a) @ normal),
                       estimated_heading_error_rad=float(wrap(pose[2] - phi)),
                       actual_heading_error_rad=float(wrap(truth[2] - phi)))
        active = valid and d.get('lqr_active', False)
        raw_wheels = np.array([v - omega*self.separation/2,
                               v + omega*self.separation/2]) / self.radius
        scale = min(1., 5.5 / max(float(np.max(np.abs(raw_wheels))), 1e-12))
        s = servo.diagnostics
        row.update(time_s=t, rate_interval_start_s=t-elapsed, control_dt_s=elapsed,
                   mission_goal_index=navigator.segment, route_id=navigator.route_id,
                   status=status, navigator_mode=navigator.mode,
                   tracker_mode=navigator.tracker.mode, planner_status=navigator.plan_status,
                   reference_valid=int(valid), lqr_active=int(active),
                   requested_v_m_s=v, raw_lqr_omega_rad_s=d.get('raw_omega', np.nan) if active else np.nan,
                   requested_omega_rad_s=omega,
                   omega_limit_active=int(active and abs(d['raw_omega']-omega) > 1e-12),
                   wheel_scale=scale, scaled_v_m_s=v*scale, scaled_omega_rad_s=omega*scale,
                   desired_left_rad_s=desired[0], desired_right_rad_s=desired[1],
                   measured_left_rad_s=measured[0], measured_right_rad_s=measured[1],
                   previous_desired_left_rad_s=self.previous_desired[0],
                   previous_desired_right_rad_s=self.previous_desired[1],
                   previous_stop=int(self.previous_stop), stop=int(stop),
                   left_servo_raw_rad_s=s['raw'][0], right_servo_raw_rad_s=s['raw'][1],
                   left_command_rad_s=commands[0], right_command_rad_s=commands[1],
                   left_servo_limit_active=int(abs(s['raw'][0]) >= 7.88-1e-12),
                   right_servo_limit_active=int(abs(s['raw'][1]) >= 7.88-1e-12),
                   left_integral_rad_s=servo.integral[0], right_integral_rad_s=servo.integral[1],
                   predicted_clearance_m=clearance,
                   est_x_m=pose[0], est_y_m=pose[1], est_heading_rad=pose[2],
                   true_x_m=truth[0], true_y_m=truth[1], true_heading_rad=truth[2])
        self.writer.writerow(row)
        self.previous_desired = np.array(desired, copy=True)
        self.previous_stop = bool(stop)
        if t >= self.next_flush:
            self.handle.flush()
            self.event_handle.flush()
            self.next_flush = t + 1.0

    def close(self):
        if not self.closed:
            self.handle.close()
            self.event_handle.close()
            self.closed = True
