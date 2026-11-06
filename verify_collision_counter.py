"""Small, separate MuJoCo fixture: prove the counter detects a known contact.

Does not load or change the mission scene. The temporary test sphere is placed
against a wall deliberately. This is a software check, not navigation evidence.
"""
from pathlib import Path
import json
import tempfile
import mujoco
from collision_counter import CollisionCounter


def main():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <geom name="wall_test" type="box" pos=".6 0 .15" size=".05 .5 .15"/>
        <body name="base" pos="0 0 .099">
          <freejoint/>
          <geom name="test_robot" type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    with tempfile.TemporaryDirectory() as folder:
        counter = CollisionCounter(model, base, folder, data.time)
        def sample_at(x):
            data.qpos[0] = x
            data.time += model.opt.timestep
            mujoco.mj_forward(model, data)
            counter.sample(data)
        sample_at(0)
        assert data.ncon > 0, 'Fixture should have a floor contact.'
        assert counter.episodes == 0, 'Floor support must not count.'
        sample_at(.5)
        assert counter.episodes == 1, 'Known wall overlap was not detected.'
        sample_at(.5)
        assert counter.episodes == 1 and counter.contact_steps == 2
        sample_at(0)
        assert not counter.active
        sample_at(.5)
        assert counter.episodes == 2
        counter.close('Fixture complete')
        report = json.loads((Path(folder)/'collision_summary.json').read_text())
        assert report['episodes'] == 2 and report['contact_steps'] == 3
        assert (Path(folder)/'collision_events.csv').is_file()
    print('PASS: floor excluded; known wall contact detected; sustained contact counted once;')
    print('      separation and re-contact counted again; final results saved.')
    print('This separate fixture does not establish that any navigation run was collision-free.')


if __name__ == '__main__':
    main()
