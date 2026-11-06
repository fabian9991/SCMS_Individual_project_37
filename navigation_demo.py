"""Terminal lifecycle for independent trials in one passive MuJoCo viewer.

Only the main thread touches MuJoCo. The reader thread queues terminal lines.
Pause freezes simulation time; reset is a new trial, never mission recovery.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time

PRESETS = {
    1: (False, False, 'No added encoder noise; normal-grip plate'),
    2: (True, False, 'Encoder noise; normal-grip plate'),
    3: (True, True, 'Encoder noise; low-grip slip plate'),
}


def apply_preset(nav, mode):
    noise, slip, _ = PRESETS[int(mode)]
    nav.ENCODER_NOISE_ENABLED = noise
    nav.SLIP_PLATE_ENABLED = slip


def load_scene(nav):
    import mujoco
    scene = nav.ROOT / 'models/robotis_tb3/scene_assignment_plate.xml'
    if not scene.is_file():
        from runpy import run_path
        run_path(str(scene.with_name('install_physical_floor.py')))['install']()
    model = mujoco.MjModel.from_xml_path(str(scene))
    return model, mujoco.MjData(model)


class TerminalCommands:
    """One daemon reader; no blocking input() in the simulation thread."""
    def __init__(self, stream=None):
        self.queue = Queue()
        self.stream = sys.stdin if stream is None else stream
        self.thread = threading.Thread(target=self._read, daemon=True,
                                       name='navigation-terminal')
        self.thread.start()

    def _read(self):
        try:
            for line in self.stream:
                self.queue.put(line.strip().lower())
        finally:
            self.queue.put('q')  # EOF: close cleanly, including redirected stdin.

    def poll(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class DemoSession:
    def __init__(self, nav, model, data, viewer, commands, mode=2, seed=37):
        import mujoco
        self.mj, self.nav = mujoco, nav
        self.model, self.data, self.viewer = model, data, viewer
        self.commands, self.mode, self.seed = commands, int(mode), int(seed)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'slip_plate')
        if min(self.floor, self.plate) < 0:
            raise ValueError('Demo requires the supplied floor and slip_plate geoms.')
        # Capture XML friction BEFORE SlipPlate can replace it with normal grip.
        self.xml_friction = model.geom_friction.copy()
        self.state = 'READY'
        self.pending = None
        self.run_dir = None
        self.start_time = 0.
        self.events = []
        self.saved_ctrl = None

    def event(self, name, **details):
        row = dict(event=name, wall_utc=datetime.now(timezone.utc).isoformat(),
                   simulation_time_s=float(self.data.time-self.start_time), **details)
        self.events.append(row)
        if self.run_dir is not None:
            with (self.run_dir/'demo_events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row)+'\n')

    def attach_run(self, run_dir, start):
        self.run_dir, self.start_time = Path(run_dir), float(start)
        self.events = []
        self.event('start', mode=self.mode, seed=self.seed,
                   description=PRESETS[self.mode][2])
        config = dict(mode=self.mode, description=PRESETS[self.mode][2], seed=self.seed,
                      encoder_noise_enabled=self.nav.ENCODER_NOISE_ENABLED,
                      slip_plate_enabled=self.nav.SLIP_PLATE_ENABLED,
                      plate_friction=self.model.geom_friction[self.plate].tolist(),
                      dock_pose_m_m_rad=self.nav.DOCK_POSE.tolist(),
                      pause_semantics='Freeze physics, sensor updates and simulation timers',
                      reset_semantics='End/save trial; restore XML initial state; fresh estimator/map/controller')
        (self.run_dir/'demo_config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')

    def select(self, mode):
        self.mode = int(mode)
        apply_preset(self.nav, self.mode)
        with self.viewer.lock():
            self.model.geom_friction[:] = self.xml_friction
            if not self.nav.SLIP_PLATE_ENABLED:
                self.model.geom_friction[self.plate] = self.xml_friction[self.floor]
        print(f'Selected {self.mode}: {PRESETS[self.mode][2]}. Seed={self.seed}.')
        print('Type s then Enter to start; q then Enter to close.', flush=True)

    def reset(self):
        """A fresh fixture at XML origin; no truth passed into estimation."""
        self.run_dir = None
        self.pending = None
        self.saved_ctrl = None
        self.events = []
        with self.viewer.lock():
            self.model.geom_friction[:] = self.xml_friction
            if not PRESETS[self.mode][1]:
                self.model.geom_friction[self.plate] = self.xml_friction[self.floor]
            self.mj.mj_resetData(self.model, self.data)
            self.viewer.user_scn.ngeom = 0
            # Same zero-command settling used by the previous standalone runner.
            for _ in range(round(1/self.model.opt.timestep)):
                self.mj.mj_step(self.model, self.data)
            self.mj.mj_forward(self.model, self.data)
        self.state = 'READY'
        self.start_time = float(self.data.time)
        apply_preset(self.nav, self.mode)
        self.viewer.sync()
        self.menu()

    def menu(self):
        print('\nDEMO READY - robot held at the launch position')
        for mode, (_, _, name) in PRESETS.items():
            print(f'  {mode}: {name}')
        target_distance = float(((self.nav.DOCK_POSE[:2]-self.nav.MARKER_XY)**2).sum()**.5)
        print(f'Current: {self.mode}; seed={self.seed}; base-to-marker target={target_distance:.3f} m.')
        print('Choose 1/2/3, then s + Enter to START. q + Enter to CLOSE.')
        print('While running: p/stop = PAUSE, s = RESUME, r = SAVE AND RESET, q = SAVE AND CLOSE.')
        print('Commands need Enter and terminal focus. h = show controls. Ctrl+C saves and closes.')
        print('Reset starts a NEW trial; pause freezes simulation time.', flush=True)

    def idle(self):
        self.viewer.sync()
        time.sleep(.02)

    def freeze(self):
        with self.viewer.lock():
            self.saved_ctrl = self.data.ctrl.copy()
            self.data.ctrl[:] = 0
        self.state = 'PAUSED'
        self.event('pause')
        print('\nPAUSED - simulation time frozen. s = resume; r = reset; q = close. Enter required.', flush=True)

    def resume(self):
        with self.viewer.lock():
            self.data.ctrl[:] = self.saved_ctrl
        self.saved_ctrl = None
        self.state = 'RUNNING'
        self.event('resume')
        print('RESUMED. p = pause; r = reset; q = close. Enter required.', flush=True)

    def before_step(self):
        """Called outside viewer.lock(), before any physics/encoder update."""
        while self.viewer.is_running():
            command = self.commands.poll()
            if command in ('q', 'quit', 'close', 'exit', 'r', 'reset'):
                self.pending = 'reset' if command in ('r', 'reset') else 'quit'
                self.event(self.pending+'_requested')
                return self.pending
            if command in ('p', 'pause', 'stop') and self.state == 'RUNNING':
                self.freeze()
            elif command in ('s', 'start', 'resume') and self.state == 'PAUSED':
                self.resume()
            elif command in ('h', 'help'):
                print('p/stop = pause; s = resume; r = save/reset; q = save/close. Add Enter.', flush=True)
            elif command not in (None, ''):
                print('Settings change only between trials. Use p, s, r or q + Enter.', flush=True)
            if self.state == 'RUNNING':
                return 'run'
            self.idle()
        self.pending = 'quit'
        self.event('viewer_closed')
        return 'quit'

    def run_finished(self, outcome):
        with self.viewer.lock():
            self.data.ctrl[:] = 0
        self.saved_ctrl = None
        self.state = 'FINISHED'
        self.event('finished', outcome=outcome)
        print('\nRUN FINISHED - final pose and trails held.')
        if self.run_dir is not None:
            print('Results folder:', self.run_dir)
        else:
            print('No run logs were opened.')
        print('r + Enter = RESET to origin and choose settings; q + Enter = CLOSE.', flush=True)

    def wait_for_start(self):
        while self.viewer.is_running():
            command = self.commands.poll()
            if command in ('q', 'quit', 'close', 'exit'):
                self.event('close_requested')
                return False
            if command in ('r', 'reset'):
                self.event('reset_requested')
                self.reset()
            elif command in ('h', 'help'):
                if self.state == 'READY':
                    self.menu()
                else:
                    print('Run finished. r + Enter = reset; q + Enter = close.', flush=True)
            elif self.state == 'READY' and command in ('1', '2', '3'):
                self.select(int(command))
            elif self.state == 'READY' and command in ('s', 'start'):
                self.state = 'RUNNING'
                print('STARTING. p = pause; r = reset; q = close. Enter required.', flush=True)
                return True
            elif command not in (None, ''):
                print('Choose 1/2/3 then s, or q.' if self.state == 'READY'
                      else 'Run finished. Use r to reset or q to close.', flush=True)
            self.idle()
        return False


def run_demo(nav, seed=None, dock_standoff=.50, view_trails=True, mode=2):
    import mujoco.viewer
    seed = nav.ENCODER_NOISE_SEED if seed is None else int(seed)
    if seed < 0:
        raise ValueError('Seed must be non-negative.')
    nav.ENCODER_NOISE_SEED = seed
    apply_preset(nav, mode)
    nav.DOCK_POSE = nav.set_dock_standoff(nav.DOCK_POSE, nav.MARKER_XY, dock_standoff)
    model, data = load_scene(nav)
    # Camera rendering dimensions must be set before constructing the viewer.
    model.vis.global_.offwidth = nav.ekf_module.WIDTH
    model.vis.global_.offheight = nav.ekf_module.HEIGHT
    commands = TerminalCommands()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        session = DemoSession(nav, model, data, viewer, commands, mode, seed)
        try:
            session.reset()
            while session.wait_for_start():
                try:
                    outcome = nav.run_mission(seed, dock_standoff, view_trails, session=session)
                except Exception as exc:
                    # The mission writer preserves errors and partial logs when available.
                    outcome = f'Error: {type(exc).__name__}: {exc}'
                    print(outcome, flush=True)
                    session.pending = None
                session.run_finished(outcome)
                if session.pending == 'quit' or not viewer.is_running():
                    break
                if session.pending == 'reset':
                    session.reset()
        except KeyboardInterrupt:
            session.event('keyboard_interrupt')
            print('\nClosing demo.', flush=True)
        finally:
            with viewer.lock():
                data.ctrl[:] = 0

