"""Evaluation-only robot/obstacle contact monitoring for the supplied MuJoCo scene.

Does not change physics, commands, planner or estimator. Sample after mj_forward
at EVERY physics step. No ground-truth quantities are returned to control.
"""
from pathlib import Path
import csv
import hashlib
import json

VERSION = 1
SUPPORT_NAMES = ('floor', 'floor_east', 'floor_south', 'floor_north', 'slip_plate')


class CollisionCounter:
    def __init__(self, model, base, run_dir, start_time):
        import mujoco
        self.model = model
        self.output = Path(run_dir)
        self.start_time = float(start_time)
        self.last_time = float(start_time)
        self.next_save = 0.0
        self.closed = False
        self.steps = self.contact_steps = self.touching_steps = 0
        self.episodes = 0
        self.contact_time = 0.0
        self.active = {}
        self.objects = {}
        self.robot_geoms = set()
        self.names = {}
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            self.names[g] = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                             or f'unnamed_geom_{g}')
            while body:
                if body == base:
                    self.robot_geoms.add(g)
                    break
                body = int(model.body_parentid[body])
        if not self.robot_geoms:
            raise ValueError('Collision counter could not identify robot geometry.')
        self.supports = {g for g, n in self.names.items()
                         if n in SUPPORT_NAMES and g not in self.robot_geoms}
        if not any(self.names[g] == 'floor' for g in self.supports):
            raise ValueError('Collision counter requires the named floor support.')
        self.file = (self.output/'collision_events.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'time_s', 'event', 'obstacle_geom', 'obstacle_geom_id',
            'robot_geom_ids', 'minimum_distance_m', 'solver_active', 'contact_episodes'])
        self.writer.writeheader()
        self.file.flush()
        self._save('RUNNING')

    def _event(self, t, event, g, info):
        self.writer.writerow(dict(time_s=t, event=event, obstacle_geom=self.names[g],
            obstacle_geom_id=g, robot_geom_ids=';'.join(map(str, sorted(info['robot_geoms']))),
            minimum_distance_m=info['distance'], solver_active=int(info['solver_active']),
            contact_episodes=self.episodes))

    def sample(self, data):
        if self.closed:
            raise RuntimeError('Cannot sample a closed collision counter.')
        now = float(data.time)
        dt = now - self.last_time
        if dt < -1e-9:
            raise ValueError('Start a new collision counter after resetting simulation time.')
        self.last_time = now
        t = now - self.start_time
        current = {}
        # Include active solver contacts conservatively, even at positive margin.
        # Also count detected touching/overlap (dist <= 0) without solver force.
        # Inactive, separated proximity records alone are not collisions.
        for i in range(data.ncon):
            con = data.contact[i]
            a, b = int(con.geom1), int(con.geom2)
            if a < 0 or b < 0 or a >= self.model.ngeom or b >= self.model.ngeom:
                continue  # supplied scene has rigid geoms, no flex contacts
            ar, br = a in self.robot_geoms, b in self.robot_geoms
            if ar == br:  # self contact or environment/environment
                continue
            robot, obstacle = (a, b) if ar else (b, a)
            if obstacle in self.supports:
                continue
            distance = float(con.dist)
            active = int(con.efc_address) >= 0
            if not active and distance > 0.0:
                continue
            info = current.setdefault(obstacle, dict(robot_geoms=set(),
                distance=distance, solver_active=False))
            info['robot_geoms'].add(robot)
            info['distance'] = min(info['distance'], distance)
            info['solver_active'] |= active
        self.steps += 1
        if current:
            if not self.active:
                self.episodes += 1
            self.contact_steps += 1
            self.contact_time += max(dt, 0.0)
            self.touching_steps += int(any(v['distance'] <= 0 for v in current.values()))
        for g, info in current.items():
            stats = self.objects.setdefault(g, dict(episodes=0, contact_steps=0,
                sampled_contact_time_s=0.0, minimum_distance_m=info['distance']))
            stats['contact_steps'] += 1
            stats['sampled_contact_time_s'] += max(dt, 0.0)
            stats['minimum_distance_m'] = min(stats['minimum_distance_m'], info['distance'])
            if g not in self.active:
                stats['episodes'] += 1
                self._event(t, 'contact_start', g, info)
                print(f'OBSTACLE CONTACT at {t:.3f}s: {self.names[g]}; '
                      f'episodes={self.episodes}; distance={info["distance"]:.6f} m')
        for g, info in self.active.items():
            if g not in current:
                self._event(t, 'contact_end', g, info)
        changed = current.keys() != self.active.keys()
        self.active = current
        if changed or t >= self.next_save:
            self.file.flush()
            self._save('RUNNING')
            self.next_save = t + 1.0

    def terminal_line(self):
        names = ', '.join(self.names[g] for g in sorted(self.active)) or 'none'
        return (f'Obstacle contacts: episodes={self.episodes}, '
                f'contact_steps={self.contact_steps}, active={names}')

    def summary_lines(self):
        result = ('NO OBSTACLE CONTACTS RECORDED' if self.episodes == 0
                  else 'OBSTACLE CONTACTS RECORDED')
        if not self.steps:
            result = 'NOT EVALUATED: no physics-step samples'
        return [
            'COLLISION MONITOR (evaluation only)', result,
            f'Obstacle-contact episodes: {self.episodes}',
            f'Physics samples with obstacle contact: {self.contact_steps}/{self.steps}',
            f'Of these, touching/overlap samples (distance <= 0): {self.touching_steps}',
            f'Sampled obstacle-contact duration: {self.contact_time:.6f} s',
            f'Monitored simulation interval: {self.last_time-self.start_time:.6f} s',
            'Excluded support geoms: '+', '.join(self.names[g] for g in sorted(self.supports)),
            'Counts robot versus external geometry; self-contact and support contact are excluded.',
            'A contact is solver-active OR has distance <= 0. Active positive-margin contacts count conservatively.',
            'An episode is consecutive sampled contact with any obstacle, separated by at least one clear sample.',
            'Zero means no qualifying contact in this monitored interval and collision model; it is not a safety proof.',
            'Interrupted runs cover only their recorded interval. Initial settling before mission start is excluded.',
        ]

    def _save(self, outcome):
        payload = dict(version=VERSION, outcome=outcome,
            collision_counter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            start_sim_time_s=self.start_time, last_sample_sim_time_s=self.last_time,
            samples=self.steps, episodes=self.episodes, contact_steps=self.contact_steps,
            touching_or_overlap_steps=self.touching_steps,
            sampled_contact_time_s=self.contact_time,
            ignored_support_geoms=[self.names[g] for g in sorted(self.supports)],
            robot_geom_ids=sorted(self.robot_geoms),
            active_obstacle_geoms=[self.names[g] for g in sorted(self.active)],
            policy='robot/external; exclude named support geoms; efc_address>=0 OR dist<=0',
            objects={self.names[g]:dict(geom_id=g, **v) for g,v in self.objects.items()})
        target = self.output/'collision_summary.json'
        temporary = self.output/'collision_summary.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
        temporary.replace(target)

    def close(self, outcome):
        if self.closed:
            return
        t = self.last_time - self.start_time
        for g, info in self.active.items():
            self._event(t, 'run_ended_while_in_contact', g, info)
        self.file.flush()
        self.file.close()
        self.closed = True
        self._save(outcome)
        (self.output/'collision_summary.txt').write_text(
            outcome+'\n'+'\n'.join(self.summary_lines())+'\n', encoding='utf-8')
