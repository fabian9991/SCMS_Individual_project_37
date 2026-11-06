"""Add pose diagnostics, viewer trails and optional docking stand-off.

Run beside the current navigate_all_goals.py after LQR and collision updates.
Backs up the file. --check validates compatibility without changing it.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

MARKER = '# NAVIGATION_VIEW_VERSION_1'


def one(items, label):
    if len(items) != 1:
        raise ValueError(f'Expected one {label}; found {len(items)}. No source changed.')
    return items[0]


def update_source(source):
    if MARKER in source:
        raise ValueError('Navigation view update is already installed.')
    if '# COLLISION_COUNTER_VERSION_1' not in source or '# LQR_DIAGNOSTICS_VERSION_1' not in source:
        raise ValueError('This update expects the earlier LQR and collision counter updates.')
    tree = ast.parse(source)
    dock = one([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Docking'], 'Docking class')
    dock_text = ast.unparse(dock)
    required_conditions = ['np.linalg.norm(base_xy - DOCK_POSE[:2]) < 0.05',
                           'np.linalg.norm(fused_pose[:2] - DOCK_POSE[:2]) < 0.03',
                           'distance < 0.02', 'self.confirmations >= 3',
                           't - self.last_seen > CAMERA_MAX_AGE']
    if not all(condition in dock_text for condition in required_conditions) or dock_text.count('np.deg2rad(3)') != 2:
        raise ValueError('Docking conditions differ from the supplied version; no files changed.')
    main = one([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main'], 'main')
    if ast.unparse(main.args) != 'seed=None':
        raise ValueError('Unexpected main arguments; expected main(seed=None).')
    def call(n):
        return ast.unparse(n.value.func) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) else ''
    def assigns(n, name):
        return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id==name for t in n.targets)
    loop = one([n for n in ast.walk(main) if isinstance(n, ast.While)], 'simulation loop')
    start = one([n for n in main.body if assigns(n, 'start')], 'mission start')
    directory = one([n for n in main.body if assigns(n, 'run_dir')], 'result directory')
    truth = one([n for n in ast.walk(loop) if assigns(n,'truth')], 'control truth sample')
    printer = one([n for n in ast.walk(loop) if isinstance(n, ast.If) and 'next_print' in ast.unparse(n.test)], 'periodic print')
    rows_if = one([n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test)=='rows'], 'results block')
    summary = one([n for n in ast.walk(main) if assigns(n, 'report') and "join(summary)" in ast.unparse(n.value)], 'summary report')
    parse = one([n for n in ast.walk(tree) if assigns(n,'args') and ast.unparse(n.value)=='parser.parse_args()'], 'CLI parse')
    invoke = one([n for n in ast.walk(tree) if call(n)=='main'], 'main invocation')
    # Inserts after a line; apply bottom-up to preserve existing source/formatting.
    inserts = [
        (directory.lineno-1, '    run_label += "_dock_default" if dock_standoff is None else ("_dock_"+format(dock_standoff, ".3f").replace(".", "p"))\n'),
        (main.lineno-1, 'from navigation_view import (ViewerTrails, StationEvaluation, pose_lines,\n'
         '                             docking_progress, set_dock_standoff, save_final_report)\n'+MARKER+'\n\n'),
        (main.lineno, '    global DOCK_POSE\n'
         '    DOCK_POSE = set_dock_standoff(DOCK_POSE, MARKER_XY, dock_standoff)\n'),
        (start.end_lineno, '    station_evaluation = StationEvaluation(model, base)\n'
         '    viewer_trails = ViewerTrails(enabled=view_trails)\n'
         '    print("VIEWER: green=true trail; blue=EKF trail; yellow=current accepted planner route; pink=desired dock base pose.")\n'
         '    print("Pose/geometry diagnostics use truth for evaluation only.")\n'
         '    print("Dock target base pose [m, m, rad]:", DOCK_POSE)\n'
         '    print("Dock success: 3 fresh accepted camera confirmations; fused XY <3 cm, visual XY <5 cm; both heading errors <3 deg.")\n'
         '    print("Dock translation stops within 2 cm of target; alignment/verification continues. Timeout: 60 s after docking starts.")\n'
         '    import json\n'
         '    import hashlib\n'
         '    (run_dir/"docking_config.json").write_text(json.dumps(dict(\n'
         '        dock_pose_m_m_rad=DOCK_POSE.tolist(), marker_xy_m=MARKER_XY.tolist(),\n'
         '        requested_standoff_m=dock_standoff, viewer_trails=view_trails,\n'
         '        target_base_to_marker_m=float(np.linalg.norm(DOCK_POSE[:2]-MARKER_XY)),\n'
         '        fused_xy_tol_m=.03, visual_xy_tol_m=.05, heading_tol_deg=3.,\n'
         '        translation_stop_m=.02, confirmation_frames=3, camera_max_age_s=CAMERA_MAX_AGE,\n'
         '        dock_timeout_s=60, diagnostic_sha256=hashlib.sha256(\n'
         '            (ROOT/"navigation_view.py").read_bytes()).hexdigest()), indent=2)+"\\n", encoding="utf-8")\n'),
        (truth.end_lineno, ' '*truth.col_offset+'viewer_trails.sample(pose, truth)\n'
         +' '*truth.col_offset+'viewer_trails.draw(viewer, tracker, DOCK_POSE)  # under viewer.lock; display only\n'),
        (printer.end_lineno, ' '*(printer.col_offset+4)+'for line in pose_lines(pose, truth, DOCK_POSE):\n'
         +' '*(printer.col_offset+8)+'print("         "+line)\n'
         +' '*(printer.col_offset+4)+'print("         "+docking_progress(docking, pose, DOCK_POSE, t, tracker.mode=="DONE", docking_started))\n'),
        (rows_if.lineno, ' '*(rows_if.col_offset+4)+'truth = evaluation_pose(data, base)  # current final physics state for reporting\n'
         +' '*(rows_if.col_offset+4)+'final_pose_lines = save_final_report(run_dir, outcome, pose, truth, DOCK_POSE, station_evaluation, data)\n'),
        (summary.lineno-1, ' '*summary.col_offset+'summary += ["", *final_pose_lines]\n'),
        (parse.lineno-1, ' '*parse.col_offset+'parser.add_argument("--dock-standoff", type=float, default=None,\n'
         +' '*(parse.col_offset+20)+'help="Optional base-to-marker target distance, 0.45..0.647 m; default retains DOCK_POSE")\n'
         +' '*parse.col_offset+'parser.add_argument("--no-trails", action="store_true", help="Disable passive-viewer trail overlays")\n'),
    ]
    lines = source.splitlines(keepends=True)
    lines[main.lineno-1] = lines[main.lineno-1].replace('main(seed=None)', 'main(seed=None, dock_standoff=None, view_trails=True)')
    lines[invoke.lineno-1:invoke.end_lineno] = [' '*invoke.col_offset+'main(seed=args.seed, dock_standoff=args.dock_standoff, view_trails=not args.no_trails)\n']
    # Main invocation is the final statement in the supported runner; guard multiline variants.
    if invoke.end_lineno != invoke.lineno:
        raise ValueError('Unexpected multiline main invocation.')
    for index, content in sorted(inserts, key=lambda item:item[0], reverse=True):
        lines.insert(index, content)
    result = ''.join(lines)
    result = result.replace('Estimated goal distance:', 'Estimated dock-target XY distance:')
    result = result.replace('Actual goal distance:', 'Actual dock-target XY distance:')
    compile(result, 'navigate_all_goals.py', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        helper = root/'navigation_view.py'
        compile(helper.read_text(encoding='utf-8-sig'), str(helper), 'exec')
        nav = one([p for p in root.iterdir() if p.is_file() and p.name.lower()=='navigate_all_goals.py'], 'navigation file')
        source = update_source(nav.read_text(encoding='utf-8-sig'))
        if args.check:
            print('Compatible. No navigation file changed. Run without --check to install.')
            return
        backup = root/('navigation_view_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir()
        shutil.copy2(nav, backup/nav.name)
        tmp = nav.with_name(nav.name+'.view_tmp')
        try:
            tmp.write_text(source, encoding='utf-8')
            tmp.replace(nav)
        finally:
            tmp.unlink(missing_ok=True)
        print('Updated:', nav.name, '\nBackup:', backup/nav.name)
        print('Default docking target, EKF, LQR, collision monitoring and planner are retained.')
        print('Use --dock-standoff 0.50 only for a separately labelled closer-docking experiment.')
    except (ValueError, SyntaxError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
