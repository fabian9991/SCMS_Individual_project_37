"""Apply diagnostics to the supplied navigation version, with backups.

Run beside navigate_all_goals.py and test_ekf.py. --check writes nothing.
All edits are prepared and syntax-checked before either source is replaced.
Existing controller, EKF, planner, seed and motion limits are preserved.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime
import os
import shutil

MARKER = '# LQR_DIAGNOSTICS_VERSION_1'


def find_file(root, name):
    matches = [p for p in root.iterdir() if p.is_file() and p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {name} beside this script; found {len(matches)}.')
    return matches[0]


def once(text, before, after):
    count = text.count(before)
    if count != 1:
        raise ValueError(f'Expected one matching code block; found {count}:\n{before[:180]}\n'
                         'No source files were changed. This file differs from the supplied version.')
    return text.replace(before, after, 1)


def update_navigation(text):
    if MARKER in text:
        raise ValueError('Diagnostics are already installed. Do not apply twice.')
    text = once(text, 'import numpy as np', '''import numpy as np
import traceback
import test_ekf as ekf_module
from lqr_diagnostics import ControllerLog, write_configuration
''' + MARKER)
    text = once(text, 'def main():', '''def main(seed=None):
    global ENCODER_NOISE_SEED
    if seed is not None:
        ENCODER_NOISE_SEED = int(seed)''')
    text = once(text, '        self.route = np.zeros((2,2))',
                '        self.route = np.zeros((2,2))\n        self.diagnostics = {}')
    text = once(text, '    def command(self, pose, endpoint_tol=0.02):',
                '    def command(self, pose, endpoint_tol=0.02):\n        self.diagnostics = {}')
    text = once(text, '        a, b = self.route[self.segment:self.segment+2]', '''        a, b = self.route[self.segment:self.segment+2]
        self.diagnostics = dict(a=a.copy(), b=b.copy(), segment=self.segment,
                                lqr_active=False)''')
    text = once(text,
                '        omega = float(np.clip(-K @ np.array([ey,eth]), -OMEGA_MAX, OMEGA_MAX))',
                '''        raw_omega = float(-K @ np.array([ey,eth]))
        omega = float(np.clip(raw_omega, -OMEGA_MAX, OMEGA_MAX))
        self.diagnostics.update(lqr_active=True, raw_omega=raw_omega)''')
    text = once(text, "        self.plan_status='waiting'", '''        self.plan_status='waiting'
        self.route_id = 0
        self.events = []''')
    text = once(text, '    def command(self,pose,ranges,t):', '''    def command(self,pose,ranges,t):
        self.tracker.diagnostics = {}''')
    text = once(text, '                self.have_path=False\n        if not self.have_path:', '''                self.events.append(dict(time_s=t, event='path_invalidated',
                                        details=dict(route_id=self.route_id,
                                                     reason='remaining path not clear')))
                self.have_path=False
        if not self.have_path:''')
    text = once(text, '            self.plan_status=self.planner.status', '''            self.plan_status=self.planner.status
            self.events.append(dict(time_s=t, event='plan_result',
                                    details=dict(goal_index=self.segment,
                                                 planner_status=self.plan_status,
                                                 path=None if path is None else path.tolist())))''')
    text = once(text, "            self.tracker.route=path;self.tracker.segment=0;self.tracker.mode='ALIGN'",
                '''            self.route_id += 1
            self.events.append(dict(time_s=t, event='route_accepted',
                                    details=dict(route_id=self.route_id, goal_index=self.segment,
                                                 path=path.tolist())))
            self.tracker.route=path;self.tracker.segment=0;self.tracker.mode='ALIGN' ''')
    text = once(text, "        if self.mode in ('DONE','OVERSHOOT'):", '''        if self.mode in ('DONE','OVERSHOOT'):
            self.events.append(dict(time_s=t,
                                    event='overshoot_replan' if self.mode=='OVERSHOOT' else 'route_complete_replan',
                                    details=dict(route_id=self.route_id,
                                                 segment=self.tracker.segment,
                                                 pose=pose.tolist())))''')
    text = once(text, '        self.integral = np.zeros(2)', '''        self.integral = np.zeros(2)
        self.diagnostics = dict(raw=np.zeros(2))''')
    text = once(text, '            self.integral[:] = 0\n            return np.zeros(2)', '''            self.integral[:] = 0
            self.diagnostics = dict(raw=np.zeros(2))
            return np.zeros(2)''')
    text = once(text, '        return np.clip(desired+0.5*error+self.integral, -7.88, 7.88)', '''        output_raw = desired+0.5*error+self.integral
        self.diagnostics = dict(raw=output_raw.copy())
        return np.clip(output_raw, -7.88, 7.88)''')
    text = once(text, "    outcome='Viewer closed'", '''    outcome='Viewer closed'
    failure = None
    diagnostics = ControllerLog(run_dir, WHEEL_RADIUS, WHEEL_SEPARATION)
    write_configuration(run_dir/'run_config.json', dict(
        encoder_noise_enabled=ENCODER_NOISE_ENABLED, encoder_noise_seed=ENCODER_NOISE_SEED,
        encoder_noise_m_per_sqrt_m=ENCODER_NOISE_M_PER_SQRT_M,
        use_wheel_travel_Q=USE_WHEEL_TRAVEL_Q, slip_plate_enabled=SLIP_PLATE_ENABLED,
        camera_R=ekf_module.R_CAMERA, camera_max_range_m=CAMERA_USE_RANGE,
        camera_dt_s=DT_CAMERA, control_dt_s=DT_CONTROL, physics_dt_s=model.opt.timestep,
        wheel_radius_m=WHEEL_RADIUS, wheel_separation_m=WHEEL_SEPARATION,
        cruise_v_m_s=V_CRUISE, omega_max_rad_s=OMEGA_MAX,
        gains=[dict(v=float(speed), K=lqr_gain(speed)[0]) for speed in tracker.speeds],
        numpy_version=np.__version__, mujoco_version=mujoco.__version__,
        note='Diagnostics only: controller and estimator settings unchanged.'),
        sources=[Path(__file__), Path(ekf_module.__file__), ROOT/'lidar_planner.py',
                 ROOT/'lqr_diagnostics.py', *sorted((ROOT/'models/robotis_tb3').glob('*.xml'))])''')
    text = once(text, '                        data.ctrl[motors]=servo.command(wheel_targets(v,omega),rates,elapsed,stop)', '''                        desired_wheels = wheel_targets(v,omega)
                        data.ctrl[motors]=servo.command(desired_wheels,rates,elapsed,stop)''')
    text = once(text, '                        truth=evaluation_pose(data,base)', '''                        truth=evaluation_pose(data,base)
                        diagnostics.record(t, elapsed, tracker, servo, pose, truth,
                                           v, omega, desired_wheels, rates,
                                           data.ctrl[motors].copy(), stop, status, clearance)''')
    text = once(text, "                        if tracker.mode=='OVERSHOOT':\n                            outcome='Stopped: segment overshoot; inspect tracking log'\n                            break\n", '')
    text = once(text, '    finally:\n        data.ctrl[motors]=0\n        camera.close()', '''    except KeyboardInterrupt:
        outcome='Interrupted by user; partial results saved'
    except Exception as exc:
        failure = exc
        outcome=f'Error: {type(exc).__name__}: {exc}; partial results saved'
        (run_dir/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        data.ctrl[motors]=0
        try:
            camera.close()
        finally:
            diagnostics.event(data.time-start, 'run_end', outcome=outcome)
            diagnostics.close()
            (run_dir/'run_status.txt').write_text(outcome+'\\n', encoding='utf-8')''')

    # Locate the broken assignment structurally. Parenthesised continuation
    # indentation is legal Python; its owning suite is the actual defect.
    tree = ast.parse(text)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'reached_bays' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Expected one reached_bays assignment. No files changed.')
    n = assignments[0]
    lines = text.splitlines(keepends=True)
    del lines[n.lineno-1:n.end_lineno]
    text = ''.join(lines)
    text = once(text, '        summary.append(\n            f"Bay arrivals declared:',
                '''        reached_bays = sum(name in recorded_goals for name in GOAL_NAMES[:-1])
        summary.append(
            f"Bay arrivals declared:''')
    # Save runtime settings instead of hard-coded summary text.
    lines = text.splitlines()
    matches = [i for i,s in enumerate(lines) if "'Camera SD settings:" in s]
    if len(matches) != 1:
        raise ValueError('Expected the existing Camera SD settings summary line.')
    lines[matches[0]] = '''                    f'Camera SD settings: range={np.sqrt(ekf_module.R_CAMERA[0,0]):g} m, '
                    f'bearing={np.degrees(np.sqrt(ekf_module.R_CAMERA[1,1])):g} deg, '
                    f'heading={np.degrees(np.sqrt(ekf_module.R_CAMERA[2,2])):g} deg.',]'''
    text = '\n'.join(lines) + '\n'
    text = once(text, "if __name__=='__main__':", '''    if not rows:
        (run_dir/'summary.txt').write_text(outcome+'\\nNo control samples recorded.\\n', encoding='utf-8')
    if failure is not None:
        raise failure

if __name__=='__main__':''')
    text = once(text, "if __name__=='__main__':\n    main()", '''if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the noise seed for this run only')
    args = parser.parse_args()
    main(seed=args.seed)''')
    compile(text, 'navigate_all_goals.py', 'exec')
    return text


def update_ekf(text):
    old = '''# Experimental standard deviations: 5 mm, 0.05 deg, 1 deg.
# Inflated above static-test errors to allow for motion and model mismatch.
'''
    if old not in text:
        return text  # Numeric settings are never modified by this installer.
    return once(text, old, '''# The tighter candidate R was tested separately. The active values below
# are the retained baseline: 2 cm range, 0.5 deg bearing, 5 deg heading.
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate edits without writing')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        for name in ('lqr_diagnostics.py', 'test_wheel_response.py'):
            if not (root/name).is_file():
                raise ValueError(f'Extract all package files first: missing {name}.')
        nav, ekf = find_file(root, 'navigate_all_goals.py'), find_file(root, 'test_ekf.py')
        updates = [(nav, update_navigation(nav.read_text(encoding='utf-8-sig'))),
                   (ekf, update_ekf(ekf.read_text(encoding='utf-8-sig')))]
        for path, source in updates:
            compile(source, str(path), 'exec')
        if args.check:
            print('Checks passed. No files changed. Run without --check to install.')
            return
        backup = root / ('lqr_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir()
        for path, source in updates:
            shutil.copy2(path, backup/path.name)
        staged = []
        try:
            for path, source in updates:
                temporary = path.with_name(path.name + '.lqr_tmp')
                temporary.write_text(source, encoding='utf-8')
                staged.append((path, temporary))
            for path, temporary in staged:
                os.replace(temporary, path)
        except Exception:
            for path, _ in updates:
                shutil.copy2(backup/path.name, path)
            raise
        finally:
            for _, temporary in staged:
                temporary.unlink(missing_ok=True)
        print('Updated:', nav.name, 'and', ekf.name)
        print('Original files backed up in:', backup)
        print('Gains, seed, EKF values, planner and motion limits are unchanged.')
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f'Update stopped: {exc}\n')


if __name__ == '__main__':
    main()
