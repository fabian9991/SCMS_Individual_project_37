"""Install contact diagnostics in the existing instrumented all-goals runner.

Run beside navigate_all_goals.py. --check previews compatibility without edits.
Requires the earlier LQR diagnostic update. Never changes controller settings.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

MARKER = '# COLLISION_COUNTER_VERSION_1'


def exactly_one(items, label):
    if len(items) != 1:
        raise ValueError(f'Expected one {label}, found {len(items)}. No source changed.')
    return items[0]


def update_source(source):
    if MARKER in source:
        raise ValueError('Collision counter already installed; do not apply twice.')
    if '# LQR_DIAGNOSTICS_VERSION_1' not in source:
        raise ValueError('Install the earlier LQR diagnostics update first. No source changed.')
    tree = ast.parse(source)
    main = exactly_one([n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'main'], 'main function')
    imports = exactly_one([n for n in tree.body if isinstance(n, ast.ImportFrom)
                           and n.module == 'lqr_diagnostics'], 'LQR diagnostic import')
    loop = exactly_one([n for n in ast.walk(main) if isinstance(n, ast.While)], 'simulation loop')
    def call_name(n):
        return ast.unparse(n.value.func) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) else ''
    forward = exactly_one([n for n in ast.walk(loop) if call_name(n)=='mujoco.mj_forward'],
                          'physics-loop mj_forward call')
    start = exactly_one([n for n in main.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id=='start' for t in n.targets)
                         and ast.unparse(n.value)=='data.time'], 'mission start assignment')
    print_if = exactly_one([n for n in ast.walk(loop) if isinstance(n, ast.If)
                            and 'next_print' in ast.unparse(n.test)], 'periodic terminal block')
    printer = exactly_one([n for n in print_if.body if call_name(n)=='print'], 'periodic status print')
    ending = exactly_one([n for n in ast.walk(main) if call_name(n)=='diagnostics.event'
                          and any(isinstance(a,ast.Constant) and a.value=='run_end'
                                  for a in n.value.args)], 'final diagnostic event')
    summary = exactly_one([n for n in ast.walk(main) if isinstance(n, ast.AugAssign)
                            and isinstance(n.target, ast.Name) and n.target.id=='summary'
                            and 'plate.summary()' in ast.unparse(n.value)], 'plate summary')
    changes = [
        (imports.end_lineno, 'from collision_counter import CollisionCounter\n'+MARKER+'\n'),
        (start.end_lineno, ' '*start.col_offset+
         'collisions = CollisionCounter(model, base, run_dir, start_time=start)\n'),
        (forward.end_lineno, ' '*forward.col_offset+'collisions.sample(data)  # evaluation only, every physics step\n'),
        (printer.end_lineno, ' '*printer.col_offset+"print('         '+collisions.terminal_line())\n"),
        (ending.lineno-1, ' '*ending.col_offset+'collisions.close(outcome)\n'),
        (summary.lineno-1, ' '*summary.col_offset+"summary += ['', *collisions.summary_lines()]\n"),
    ]
    lines = source.splitlines(keepends=True)
    for i, content in sorted(changes, reverse=True):
        lines.insert(i, content)
    result = ''.join(lines)
    result = result.replace('No collision detection metric is included.',
        'Obstacle contacts are monitored at each physics step; see collision_summary.txt.')
    compile(result, 'navigate_all_goals.py', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        counter = root/'collision_counter.py'
        if not counter.is_file():
            raise ValueError('Copy collision_counter.py beside this installer first.')
        compile(counter.read_text(encoding='utf-8-sig'), str(counter), 'exec')
        nav = exactly_one([p for p in root.iterdir() if p.is_file()
                           and p.name.lower()=='navigate_all_goals.py'], 'navigation file')
        result = update_source(nav.read_text(encoding='utf-8-sig'))
        if args.check:
            print('Compatible. No files changed. Run without --check to install.')
            return
        backup = root/('collision_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir()
        shutil.copy2(nav, backup/nav.name)
        temporary = nav.with_name(nav.name+'.collision_tmp')
        try:
            temporary.write_text(result, encoding='utf-8')
            temporary.replace(nav)
        finally:
            temporary.unlink(missing_ok=True)
        print('Updated:', nav.name)
        print('Backup:', backup/nav.name)
        print('Obstacle contacts will appear in the terminal and result files.')
        print('The patch adds evaluation only; controller and estimator settings are retained.')
    except (ValueError, SyntaxError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
