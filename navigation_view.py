"""Evaluation-only pose reporting and passive-viewer overlays.
No values from this module may be passed into control, planning or the EKF.
"""
import json
from pathlib import Path
import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi


def error(a, b):
    result = np.asarray(a, float) - np.asarray(b, float)
    result[2] = wrap(result[2])
    return result


def pose_text(p):
    return f'x={p[0]:+.4f} m, y={p[1]:+.4f} m, theta={np.degrees(wrap(p[2])):+.3f} deg'


def error_text(e):
    return (f'dx={100*e[0]:+.2f} cm, dy={100*e[1]:+.2f} cm, '
            f'dtheta={np.degrees(e[2]):+.3f} deg, XY={100*np.linalg.norm(e[:2]):.2f} cm')


def pose_lines(estimate, truth, target):
    return ['ESTIMATED BASE: '+pose_text(estimate),
            'TRUE BASE (evaluation only): '+pose_text(truth),
            'LOCALISATION (estimate - truth): '+error_text(error(estimate, truth)),
            'DESIRED DOCK BASE: '+pose_text(target),
            'ESTIMATED DOCK ERROR (estimate - target): '+error_text(error(estimate, target)),
            'ACTUAL DOCK ERROR (truth - target): '+error_text(error(truth, target))]


def docking_progress(docking, pose, target, t, active, started):
    if not active:
        return 'DOCK: not active; dock errors above refer to the future docking target.'
    age = t-docking.last_seen
    age_text = f'{age:.2f} s' if np.isfinite(age) else 'never accepted'
    visual = 'none' if docking.visual_pose is None else error_text(error(docking.visual_pose, target))
    elapsed = 0. if started is None else t-started
    return (f'DOCK: {docking.mode}; accepted-camera age={age_text}; '
            f'confirmations={docking.confirmations}/3; elapsed={elapsed:.2f}/60 s; '
            f'last visual pose - target: {visual}')


def set_dock_standoff(original, marker_xy, metres):
    """Optional known-map target choice. Never reads simulation truth.

    0.45 m is an experimental lower limit for this supplied
    camera/panel geometry, not a guarantee of visibility or collision clearance.
    """
    if metres is None:
        return np.asarray(original, float).copy()
    if not np.isfinite(metres) or not 0.45 <= metres <= 0.647:
        raise ValueError('--dock-standoff must be between 0.45 and 0.647 m for this scene.')
    heading = original[2]
    return np.array([*(np.asarray(marker_xy)-metres*np.array([np.cos(heading), np.sin(heading)])), heading])


class StationEvaluation:
    """Reads the actual station geometry only for display/scoring."""
    def __init__(self, model, base):
        import mujoco
        self.mj, self.model = mujoco, model
        self.charger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'charger')
        self.marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'charger_marker')
        if min(self.charger, self.marker) < 0:
            raise ValueError('Station diagnostics require charger and charger_marker geoms.')
        if model.geom_type[self.charger] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('Station diagnostics expect the supplied box charger.')
        bodies = set()
        for body in range(model.nbody):
            parent = body
            while parent:
                if parent == base:
                    bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.robot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                            and (model.geom_contype[g] or model.geom_conaffinity[g])]

    def measure(self, data, truth, target):
        model = self.model
        center = data.geom_xpos[self.charger].copy()
        rotation = data.geom_xmat[self.charger].reshape(3, 3)
        normal = data.geom_xmat[self.marker].reshape(3, 3)[:, 2].copy()
        normal /= np.linalg.norm(normal)
        # Choose the charger box face pointing in the panel's front direction.
        local_normal = rotation.T @ normal
        axis = int(np.argmax(np.abs(local_normal)))
        face_normal = rotation[:, axis] * np.sign(local_normal[axis])
        face = center + model.geom_size[self.charger, axis]*face_normal
        panel = data.geom_xpos[self.marker] + model.geom_size[self.marker, 2]*normal
        n = face_normal[:2] / np.linalg.norm(face_normal[:2])
        tangent = np.array([-n[1], n[0]])
        delta = truth[:2]-face[:2]
        # Signed surface distance in MuJoCo collision geometry, not centre distance.
        distances = [float(self.mj.mj_geomDistance(model, data, g, self.charger, 20., None))
                     for g in self.robot_geoms]
        gap = min(distances) if distances else float('nan')
        facing = float(wrap(np.arctan2(n[1], n[0])+np.pi))
        return dict(station_facing_heading_rad=facing,
                    actual_heading_error_to_station_deg=float(np.degrees(wrap(truth[2]-facing))),
                    charger_center_world_m=center.tolist(), charger_front_world_m=face.tolist(),
                    marker_front_world_m=panel.tolist(),
                    actual_base_to_marker_xy_m=float(np.linalg.norm(truth[:2]-panel[:2])),
                    actual_base_to_charger_front_normal_m=float(delta@n),
                    actual_base_lateral_to_charger_m=float(delta@tangent),
                    desired_base_to_charger_front_normal_m=float((target[:2]-face[:2])@n),
                    robot_to_charger_collision_surface_distance_m=gap if gap < 20 else None)

    def lines(self, data, truth, target):
        m = self.measure(data, truth, target)
        gap = m['robot_to_charger_collision_surface_distance_m']
        return ['STATION GEOMETRY (evaluation only; world metres):',
                f"  Charger centre XYZ={np.round(m['charger_center_world_m'], 4).tolist()}",
                f"  Charger front XYZ={np.round(m['charger_front_world_m'], 4).tolist()}; "
                f"marker front XYZ={np.round(m['marker_front_world_m'], 4).tolist()}",
                f"  Face-aligned robot heading={np.degrees(m['station_facing_heading_rad']):+.3f} deg; "
                f"actual heading error to station={m['actual_heading_error_to_station_deg']:+.3f} deg",
                f"  Base-to-marker XY distance={m['actual_base_to_marker_xy_m']:.4f} m",
                f"  Base-to-charger front normal gap={m['actual_base_to_charger_front_normal_m']:.4f} m "
                f"(target {m['desired_base_to_charger_front_normal_m']:.4f} m); "
                f"lateral offset={m['actual_base_lateral_to_charger_m']:+.4f} m",
                '  Robot collision-surface to charger gap='+('unavailable' if gap is None else f'{gap:.4f} m'),
                '  Centre/face distances are not body clearance. Surface gap uses simulated collision geometry.']


class ViewerTrails:
    """Viewer-only lines; never enter camera images, LiDAR rays or contacts."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.estimated, self.actual = [], []
        self.warned = False

    def sample(self, estimate, truth):
        for track, pose in ((self.estimated, estimate), (self.actual, truth)):
            p = np.asarray(pose[:2], float).copy()
            if not track or np.linalg.norm(p-track[-1]) >= .015:
                track.append(p)
                if len(track) > 600:
                    track[:] = track[::2] + ([track[-1]] if len(track)%2 == 0 else [])

    def draw(self, viewer, navigator, target):
        if not self.enabled:
            return
        import mujoco
        scene = getattr(viewer, 'user_scn', None)
        if scene is None:
            if not self.warned:
                print('Viewer trails unavailable: this viewer has no user_scn.')
                self.warned = True
            return
        scene.ngeom = 0
        def line(a, b, color, z, radius=.007):
            if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(b)-a) < 1e-8:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color, np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                 np.array([*a, z]), np.array([*b, z]))
            scene.ngeom += 1
        # Draw full currently accepted route (not a goal-to-goal straight shortcut).
        # Clear it immediately if invalidated or docking replaces route tracking.
        if navigator.have_path and navigator.mode != 'DONE':
            path = navigator.tracker.route
            for a, b in zip(path[:-1], path[1:]):
                line(a, b, (1., .8, .05, 1.), .04)
        for track, color, z in ((self.actual, (.1, 1., .2, 1.), .055),
                                 (self.estimated, (.1, .55, 1., 1.), .07)):
            # Reserve scene capacity evenly; preserve the entire trail by decimation.
            limit = max(2, (scene.maxgeom-scene.ngeom-5)//2)
            indices = np.unique(np.linspace(0, len(track)-1, min(len(track), limit)).astype(int))
            points = [track[i] for i in indices]
            for a, b in zip(points[:-1], points[1:]):
                line(a, b, color, z)
        p = np.asarray(target[:2])
        line(p+[-.045, 0], p+[.045, 0], (1., .2, .8, 1.), .085)
        line(p+[0, -.045], p+[0, .045], (1., .2, .8, 1.), .085)
        line(p, p+.18*np.array([np.cos(target[2]), np.sin(target[2])]), (1., .2, .8, 1.), .085)


def save_final_report(folder, outcome, estimate, truth, target, station, data):
    lines = ['FINAL POSE AND DOCKING REPORT', 'Outcome: '+outcome,
             'All theta values below are degrees, wrapped to [-180, 180).',
             *pose_lines(estimate, truth, target), *station.lines(data, truth, target)]
    report = '\n'.join(lines)+'\n'
    Path(folder, 'final_pose_report.txt').write_text(report, encoding='utf-8')
    payload = dict(outcome=outcome, estimated_pose_m_m_rad=np.asarray(estimate).tolist(),
                   true_pose_m_m_rad=np.asarray(truth).tolist(), desired_dock_pose_m_m_rad=np.asarray(target).tolist(),
                   localisation_error_m_m_rad=error(estimate, truth).tolist(),
                   estimated_dock_error_m_m_rad=error(estimate, target).tolist(),
                   actual_dock_error_m_m_rad=error(truth, target).tolist(),
                   station=station.measure(data, truth, target))
    Path(folder, 'final_pose_report.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print('\n'+report)
    return lines
