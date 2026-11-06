r"""Measure the existing wheel servo without changing any gains.

Run beside the navigation script: .venv\Scripts\python.exe test_wheel_response.py
Optional --viewer displays the tests. Default runs without rendering.
Each short case resets to the XML starting pose, settles, drives, then stops.
No EKF or LQR is exercised here. The mission's wheel servo already uses ideal
encoder rates, so this test uses that same input. Mission encoder noise stays on.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import time
import traceback
import numpy as np
from lqr_diagnostics import write_configuration

ROOT = Path(__file__).resolve().parent
CASES = [
    ('straight_slow', .025, 0., 3.),
    ('straight_medium', .10, 0., 3.),
    ('straight_cruise', .16, 0., 3.),
    ('turn_left', 0., .50, 2.5),
    ('turn_right', 0., -.50, 2.5),
    ('curve_left', .10, .30, 3.),
    ('curve_right', .10, -.30, 3.),
]
STOP_TIME = 2.
FIELDS = ['case', 'time_s', 'interval_start_s', 'control_dt_s', 'phase',
          'desired_left_rad_s', 'desired_right_rad_s',
          'measured_left_rad_s', 'measured_right_rad_s',
          'left_error_rad_s', 'right_error_rad_s',
          'applied_left_command_rad_s', 'applied_right_command_rad_s',
          'next_desired_left_rad_s', 'next_desired_right_rad_s',
          'next_left_command_rad_s', 'next_right_command_rad_s',
          'next_left_raw_rad_s', 'next_right_raw_rad_s',
          'next_left_limit_active', 'next_right_limit_active',
          'left_integral_rad_s', 'right_integral_rad_s',
          'actual_forward_speed_m_s', 'actual_yaw_rate_rad_s']


def navigation_module():
    files = [p for p in ROOT.iterdir() if p.name.lower() == 'navigate_all_goals.py']
    if len(files) != 1:
        raise RuntimeError('Place this file beside your single navigate_all_goals.py.')
    spec = importlib.util.spec_from_file_location('wheel_test_navigation', files[0])
    nav = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nav)  # Guarded main() is not run; camera is not instantiated.
    if not hasattr(nav.WheelServo(), 'diagnostics'):
        raise RuntimeError('Run apply_lqr_updates.py before this test.')
    return nav


def summarise(rows, outcomes, output):
    result = []
    for name, v, omega, drive_time in CASES:
        samples = [r for r in rows if r['case'] == name]
        if not samples:
            continue
        drive = [r for r in samples if r['phase'] == 'DRIVE']
        late = [r for r in drive if r['interval_start_s'] >= drive_time-1.]
        stop = [r for r in samples if r['phase'] == 'STOP'
                and r['interval_start_s'] >= drive_time+.5]
        record = dict(case=name, outcome=outcomes.get(name, 'partial'),
                      requested_v_m_s=v, requested_omega_rad_s=omega,
                      drive_samples=len(drive), late_drive_samples=len(late))
        for side in ('left', 'right'):
            errors = np.array([r[f'{side}_error_rad_s'] for r in drive])
            weights = np.array([r['control_dt_s'] for r in drive])
            record[f'{side}_drive_error_rms_rad_s'] = (
                float(np.sqrt(np.average(errors**2, weights=weights))) if len(errors) else np.nan)
            record[f'{side}_late_mean_error_rad_s'] = (
                float(np.mean([r[f'{side}_error_rad_s'] for r in late])) if late else np.nan)
            record[f'{side}_late_measured_mean_rad_s'] = (
                float(np.mean([r[f'measured_{side}_rad_s'] for r in late])) if late else np.nan)
            record[f'{side}_stop_max_abs_rate_after_0_5_s'] = (
                max(abs(r[f'measured_{side}_rad_s']) for r in stop) if stop else np.nan)
            record[f'{side}_limit_samples'] = sum(r[f'next_{side}_limit_active'] for r in samples)
        result.append(record)
    if result:
        with (output/'wheel_response_summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(result[0]))
            w.writeheader()
            w.writerows(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    import mujoco
    nav = navigation_module()
    scene = ROOT/'models/robotis_tb3/scene_assignment_plate.xml'
    if not scene.is_file():
        raise FileNotFoundError(f'Missing mission scene: {scene}')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    def required(kind, name):
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            raise ValueError(f'Missing {name}')
        return i

    floor = required(mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    plate = required(mujoco.mjtObj.mjOBJ_GEOM, 'slip_plate')
    model.geom_friction[plate] = model.geom_friction[floor]  # Same plate-off setting as mission.
    base = required(mujoco.mjtObj.mjOBJ_BODY, 'base')
    joints = [required(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ('wheel_left', 'wheel_right')]
    addresses = model.jnt_qposadr[joints]
    motors = [required(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ('wheel_left', 'wheel_right')]
    output = ROOT/'results'/('wheel_response_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    write_configuration(output/'test_config.json', dict(
        test='Current wheel servo, no controller tuning', cases=CASES,
        stop_time_s=STOP_TIME, control_dt_s=nav.DT_CONTROL,
        physics_dt_s=model.opt.timestep, slip_plate_enabled=False,
        feedback='Ideal wheel encoder rates, matching mission WheelServo',
        encoder_noise_in_this_test=False,
        note='Noise is injected only into the mission EKF, which this inner-loop test does not run.',
        mujoco_version=mujoco.__version__),
        sources=[Path(__file__), Path(nav.__file__), *sorted(scene.parent.glob('*.xml'))])
    print('Output folder:', output)
    print('Testing existing servo; each case starts at the XML pose. No gains change.')
    viewer = None
    rows, outcomes = [], {}
    outcome = 'COMPLETED'
    failure = None
    try:
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
        with (output/'wheel_response.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for name, v, omega, drive_time in CASES:
                if viewer is not None and not viewer.is_running():
                    outcome = 'VIEWER CLOSED; partial results'
                    break
                # Resets are test setup only. No simulator pose controls commands.
                with viewer.lock() if viewer is not None else nullcontext():
                    mujoco.mj_resetData(model, data)
                    for _ in range(round(1./model.opt.timestep)):
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    servo = nav.WheelServo()
                    start = last_control = data.time
                    enc_previous = data.qpos[addresses].copy()
                    previous_pose = nav.evaluation_pose(data, base)
                    desired_previous = np.zeros(2)
                    command_previous = np.zeros(2)
                    phase_previous = 'INITIAL'
                    next_control = 0.
                outcomes[name] = 'partial'
                print('Testing', name)
                while data.time-start < drive_time+STOP_TIME:
                    if viewer is not None and not viewer.is_running():
                        outcome = 'VIEWER CLOSED; partial results'
                        break
                    wall = time.perf_counter()
                    # Viewer locking is scoped to physics and state changes.
                    with viewer.lock() if viewer is not None else nullcontext():
                        mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        t = data.time-start
                        if t >= next_control:
                            dt = data.time-last_control
                            enc = data.qpos[addresses].copy()
                            rates = (enc-enc_previous)/dt
                            desired = nav.wheel_targets(v, omega) if t < drive_time else np.zeros(2)
                            stop = t >= drive_time
                            commands = servo.command(desired, rates, dt, stop=stop)
                            data.ctrl[motors] = commands
                            # Actual movement below is evaluation only.
                            truth = nav.evaluation_pose(data, base)
                            dtheta = nav.wrap_angle(truth[2]-previous_pose[2])
                            mid = previous_pose[2]+dtheta/2
                            forward_speed = float((truth[:2]-previous_pose[:2]) @
                                                  np.array([np.cos(mid), np.sin(mid)]))/dt
                            row = dict(case=name, time_s=t, interval_start_s=t-dt,
                                       control_dt_s=dt, phase=phase_previous,
                                       actual_forward_speed_m_s=forward_speed,
                                       actual_yaw_rate_rad_s=dtheta/dt)
                            for i, side in enumerate(('left', 'right')):
                                row.update({f'desired_{side}_rad_s': desired_previous[i],
                                            f'measured_{side}_rad_s': rates[i],
                                            f'{side}_error_rad_s': desired_previous[i]-rates[i],
                                            f'applied_{side}_command_rad_s': command_previous[i],
                                            f'next_desired_{side}_rad_s': desired[i],
                                            f'next_{side}_command_rad_s': commands[i],
                                            f'next_{side}_raw_rad_s': servo.diagnostics['raw'][i],
                                            f'next_{side}_limit_active': int(abs(servo.diagnostics['raw'][i]) >= 7.88-1e-12),
                                            f'{side}_integral_rad_s': servo.integral[i]})
                            writer.writerow(row)
                            handle.flush()
                            rows.append(row)
                            previous_pose, enc_previous = truth, enc
                            desired_previous, command_previous = desired.copy(), commands.copy()
                            phase_previous = 'STOP' if stop else 'DRIVE'
                            last_control = data.time
                            next_control = t + nav.DT_CONTROL
                    if viewer is not None:
                        viewer.sync()
                        delay = model.opt.timestep-(time.perf_counter()-wall)
                        if delay > 0:
                            time.sleep(delay)
                data.ctrl[motors] = 0
                if outcome != 'COMPLETED':
                    break
                outcomes[name] = 'completed'
    except KeyboardInterrupt:
        outcome = 'INTERRUPTED; partial results'
    except Exception as exc:
        failure = exc
        outcome = f'ERROR: {type(exc).__name__}: {exc}; partial results'
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        data.ctrl[motors] = 0
        if viewer is not None:
            viewer.close()
        summarise(rows, outcomes, output)
        (output/'summary.txt').write_text(outcome+'\n'+json.dumps(outcomes, indent=2)+'\n', encoding='utf-8')
        print(outcome)
        print('Results:', output)
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
