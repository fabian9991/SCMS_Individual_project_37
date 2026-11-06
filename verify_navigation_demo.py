"""Non-visual lifecycle checks using the project's real MuJoCo model.

Does not start a navigation trial or change the user's result folders.
Checks reset determinism, friction switching and pause/resume physics state.
The Windows terminal and visible viewer still need the short manual check.
"""
from contextlib import nullcontext
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import navigate_all_goals as nav
from navigation_demo import DemoSession, TerminalCommands, load_scene


class ViewerProbe:
    def __init__(self):
        self.user_scn = SimpleNamespace(ngeom=0)
        self.running = True
        self.sync_count = 0

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.running

    def sync(self):
        self.sync_count += 1


class Commands:
    def __init__(self, values=()):
        self.values = list(values)

    def poll(self):
        return self.values.pop(0) if self.values else None


def verify():
    import mujoco
    model, data = load_scene(nav)
    viewer = ViewerProbe()
    session = DemoSession(nav, model, data, viewer, Commands(), mode=2, seed=37)
    session.reset()
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    initial_time = data.time
    xml_mu = session.xml_friction[session.plate].copy()
    floor_mu = session.xml_friction[session.floor].copy()
    assert not np.array_equal(xml_mu, floor_mu), 'Expected a genuinely lower-grip insert.'
    for mode in (3, 1, 2, 3):
        session.select(mode)
        expected = xml_mu if mode == 3 else floor_mu
        np.testing.assert_array_equal(model.geom_friction[session.plate], expected)
        assert nav.ENCODER_NOISE_ENABLED == (mode != 1)
        assert nav.SLIP_PLATE_ENABLED == (mode == 3)
    session.select(2)
    session.state = 'RUNNING'
    data.ctrl[:] = .4
    saved_ctrl = data.ctrl.copy()
    qpos, qvel, sim_time = data.qpos.copy(), data.qvel.copy(), data.time
    session.freeze()
    for _ in range(3):
        viewer.sync()  # No stepping while paused.
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == sim_time and not np.any(data.ctrl)
    session.resume()
    np.testing.assert_array_equal(data.ctrl, saved_ctrl)
    assert session.state == 'RUNNING'
    # User pause then resume is processed before another physics step.
    session.commands = Commands(['p', 's', 'r'])
    assert session.before_step() == 'run'
    assert data.time == sim_time
    assert session.before_step() == 'reset'
    # Disturb the fixture, then verify a full reset and cleared overlays.
    for _ in range(20):
        mujoco.mj_step(model, data)
    viewer.user_scn.ngeom = 123
    session.reset()
    np.testing.assert_array_equal(data.qpos, initial_qpos)
    np.testing.assert_array_equal(data.qvel, initial_qvel)
    assert data.time == initial_time and viewer.user_scn.ngeom == 0
    assert not np.any(data.ctrl) and session.run_dir is None
    # Completion holds the same data and overlays; start cannot restart it.
    with tempfile.TemporaryDirectory() as tmp:
        session.attach_run(Path(tmp), data.time)
        viewer.user_scn.ngeom = 15
        qpos = data.qpos.copy()
        session.run_finished('Verification fixture complete')
        session.commands = Commands(['s', 'q'])
        assert not session.wait_for_start()
        assert viewer.is_running() and viewer.user_scn.ngeom == 15
        np.testing.assert_array_equal(data.qpos, qpos)
        assert (Path(tmp)/'demo_config.json').is_file()
        assert 'finished' in (Path(tmp)/'demo_events.jsonl').read_text()
    terminal = TerminalCommands(io.StringIO('1\ns\np\ns\nr\nq\n'))
    terminal.thread.join(timeout=2)
    assert [terminal.poll() for _ in range(7)] == ['1','s','p','s','r','q','q']
    assert np.isclose(np.linalg.norm(nav.DOCK_POSE[:2]-nav.MARKER_XY), .50)
    print('\nPASS: defaults, presets/friction restore, reset, pause/resume, completion hold, terminal parsing.')
    print('This is a lifecycle check; run the manual viewer check before recording the demo.')


if __name__ == '__main__':
    verify()
