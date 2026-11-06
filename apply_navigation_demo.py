"""Install terminal demo controls on the navigation-view version of the runner.

Run --check first. Makes a timestamped backup and refuses unexpected source.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

MARKER = '# NAVIGATION_DEMO_VERSION_1'


def update_source(source):
    if MARKER in source:
        raise ValueError('Demo controls are already installed; no file changed.')
    for required in ('# NAVIGATION_VIEW_VERSION_1', '# COLLISION_COUNTER_VERSION_1',
                     '# LQR_DIAGNOSTICS_VERSION_1'):
        if required not in source:
            raise ValueError('Install the previous navigation-view/collision/LQR updates first.')
    ast.parse(source)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Source differs from the supported runner near: '+old[:90])
        source = source.replace(old, new, 1)

    replace('Routes are generated from LiDAR observations. No XML changes or teleportation.',
            'Routes use LiDAR observations. Demo reset restores the XML start between trials.')
    replace('from pathlib import Path', 'from pathlib import Path\nfrom contextlib import nullcontext')
    replace('# Marker-to-base separation .647 m; camera-to-marker approximately .557 m.\nDOCK_POSE = np.array([-8.05, 0., np.pi])',
            '# Default marker-to-base separation .50 m; camera-to-marker about .41 m.\n'
            'DEFAULT_DOCK_STANDOFF = 0.50\n'
            'DOCK_POSE = np.array([MARKER_XY[0]+DEFAULT_DOCK_STANDOFF, MARKER_XY[1], np.pi])')
    replace('def main(seed=None, dock_standoff=None, view_trails=True):',
            MARKER+'\n\ndef run_mission(seed=None, dock_standoff=DEFAULT_DOCK_STANDOFF, view_trails=True, session=None):')
    replace('    DOCK_POSE = set_dock_standoff(DOCK_POSE, MARKER_XY, dock_standoff)',
            '    dock_standoff = DEFAULT_DOCK_STANDOFF if dock_standoff is None else dock_standoff\n'
            '    DOCK_POSE = set_dock_standoff(DOCK_POSE, MARKER_XY, dock_standoff)')
    replace("    scene = ROOT / 'models/robotis_tb3/scene_assignment_plate.xml'\n"
            "    if not scene.is_file():\n"
            "        from runpy import run_path\n"
            "        run_path(str(scene.with_name('install_physical_floor.py')))['install']()\n"
            '    model = mujoco.MjModel.from_xml_path(str(scene))\n'
            '    data = mujoco.MjData(model)',
            '    from navigation_demo import load_scene\n'
            '    model, data = load_scene(__import__(__name__)) if session is None else (session.model, session.data)')
    # __import__ on __main__ and this flat module is supported; use sys.modules for clarity.
    replace('    model, data = load_scene(__import__(__name__)) if session is None else (session.model, session.data)',
            '    import sys\n'
            '    model, data = load_scene(sys.modules[__name__]) if session is None else (session.model, session.data)')
    replace('    for _ in range(round(1/model.opt.timestep)):\n'
            '        mujoco.mj_step(model,data)\n'
            '    mujoco.mj_forward(model,data)',
            '    if session is None:\n'
            '        for _ in range(round(1/model.opt.timestep)):\n'
            '            mujoco.mj_step(model,data)\n'
            '        mujoco.mj_forward(model,data)')
    replace("    plate = SlipPlate(model,wheel_bodies,floor,required(mujoco.mjtObj.mjOBJ_GEOM,'slip_plate'))",
            '    with nullcontext() if session is None else session.viewer.lock():\n'
            "        plate = SlipPlate(model,wheel_bodies,floor,required(mujoco.mjtObj.mjOBJ_GEOM,'slip_plate'))")
    replace('    camera = MarkerCamera(model)', '    camera = None  # allocated inside try so camera errors also save partial results')
    replace("        note='Diagnostics only: controller and estimator settings unchanged.'),",
            "        demo_mode=None if session is None else session.mode,\n"
            "        dock_standoff_m=dock_standoff,\n"
            "        note='Terminal lifecycle added; 0.50 m default stand-off; EKF/LQR/planner unchanged.'),")
    replace("                 ROOT/'lqr_diagnostics.py', *sorted((ROOT/'models/robotis_tb3').glob('*.xml'))])",
            "                 ROOT/'lqr_diagnostics.py', ROOT/'navigation_demo.py', ROOT/'navigation_view.py',\n"
            "                 ROOT/'collision_counter.py', *sorted((ROOT/'models/robotis_tb3').glob('*.xml'))])")
    replace("    print('Output folder:', run_dir)",
            "    if session is not None:\n"
            "        session.attach_run(run_dir, start)\n"
            "    print('Output folder:', run_dir)")
    replace('    try:\n        with mujoco.viewer.launch_passive(model,data) as viewer:\n'
            '            while viewer.is_running():\n'
            '                wall=time.perf_counter()',
            '    try:\n'
            '        with nullcontext() if session is None else session.viewer.lock():\n'
            '            camera = MarkerCamera(model)\n'
            '        viewer_context = (mujoco.viewer.launch_passive(model,data) if session is None\n'
            '                          else nullcontext(session.viewer))\n'
            '        with viewer_context as viewer:\n'
            '            while viewer.is_running():\n'
            '                if session is not None:\n'
            '                    action = session.before_step()\n'
            '                    if action != "run":\n'
            '                        outcome = ("Reset requested; partial results saved" if action == "reset"\n'
            '                                   else "Closed by user; partial results saved")\n'
            '                        break\n'
            '                wall=time.perf_counter()')
    replace('                            print("         "+docking_progress(docking, pose, DOCK_POSE, t, tracker.mode=="DONE", docking_started))',
            '                            sigma = np.sqrt(np.maximum(np.diag(P), 0.))\n'
            '                            print(f"         EKF 1-sigma: x={100*sigma[0]:.2f} cm, y={100*sigma[1]:.2f} cm, theta={np.degrees(sigma[2]):.2f} deg")\n'
            '                            print("         "+docking_progress(docking, pose, DOCK_POSE, t, tracker.mode=="DONE", docking_started))')
    replace("    except KeyboardInterrupt:\n        outcome='Interrupted by user; partial results saved'",
            "    except KeyboardInterrupt:\n        outcome='Interrupted by user; partial results saved'\n"
            "        if session is not None:\n            session.pending = 'quit'\n"
            "            session.event('keyboard_interrupt')")
    replace('        data.ctrl[motors]=0\n        try:\n            camera.close()',
            '        if session is None:\n'
            '            data.ctrl[motors]=0\n'
            '        else:\n'
            '            with session.viewer.lock():\n'
            '                data.ctrl[motors]=0\n'
            '        try:\n'
            '            if camera is not None:\n'
            '                camera.close()')
    replace('    if failure is not None:\n        raise failure',
            '    if failure is not None:\n        raise failure\n    return outcome')
    replace("if __name__=='__main__':", '''def main(seed=None, dock_standoff=DEFAULT_DOCK_STANDOFF, view_trails=True, mode=2, auto=False):
    import sys
    from navigation_demo import apply_preset, run_demo
    if seed is not None and int(seed) < 0:
        raise ValueError('Seed must be non-negative.')
    apply_preset(sys.modules[__name__], mode)
    if auto:
        return run_mission(seed, dock_standoff, view_trails)
    return run_demo(sys.modules[__name__], seed, dock_standoff, view_trails, mode)


if __name__=='__main__':''')
    replace('parser.add_argument("--dock-standoff", type=float, default=None,',
            'parser.add_argument("--dock-standoff", type=float, default=DEFAULT_DOCK_STANDOFF,')
    replace('help="Optional base-to-marker target distance, 0.45..0.647 m; default retains DOCK_POSE"',
            'help="Base-to-marker target distance, 0.45..0.647 m; default 0.50 m"')
    replace('    args = parser.parse_args()',
            '    parser.add_argument("--mode", type=int, choices=(1,2,3), default=2, help="Initial demo preset: 1 clean, 2 noise, 3 noise + slip")\n'
            '    parser.add_argument("--auto", action="store_true", help="Run once immediately without menu; close viewer at end")\n'
            '    args = parser.parse_args()')
    replace('    main(seed=args.seed, dock_standoff=args.dock_standoff, view_trails=not args.no_trails)',
            '    main(seed=args.seed, dock_standoff=args.dock_standoff, view_trails=not args.no_trails, mode=args.mode, auto=args.auto)')
    compile(source, 'navigate_all_goals.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        for name in ('navigation_demo.py', 'navigation_view.py'):
            compile((root/name).read_text(encoding='utf-8-sig'), name, 'exec')
        target = root/'navigate_all_goals.py'
        source = update_source(target.read_text(encoding='utf-8-sig'))
        if args.check:
            print('Compatible. No navigation file changed. Run without --check to install.')
            return
        backup = root/('navigation_demo_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir()
        shutil.copy2(target, backup/target.name)
        temp = target.with_name(target.name+'.demo_tmp')
        try:
            temp.write_text(source, encoding='utf-8')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        print('Installed terminal demo controls and 0.50 m default stand-off.')
        print('Backup:', backup/target.name)
        print('Run navigate_all_goals.py, choose 1/2/3 then s + Enter.')
    except (OSError, SyntaxError, ValueError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
