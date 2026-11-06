"""Track A full-route preview: estimated-state LQR tracking with LiDAR stop.
Place beside the supplied orientation-aware test_ekf.py. Its main() is NOT run.
Run: python navigate_all_goals.py
Routes are generated from LiDAR observations. No XML changes or teleportation.
All dynamics/controller parameters are provisional, documented below.
"""
from pathlib import Path
import csv
from datetime import datetime
import time
import numpy as np
from lidar_planner import LidarPlanner
from test_ekf import (MarkerCamera, predict_ekf, correct_ekf, motion,
                      wrap_angle, WHEEL_RADIUS, WHEEL_SEPARATION, MARKER_XY, CAMERA_OFFSET)

ROOT = Path(__file__).resolve().parent
DT_CONTROL = 0.05
DT_CAMERA = 0.5
MAX_TIME = 900.0
# Declared nominal launch pose from supplied XML, NOT read from simulation truth.
START_ESTIMATE = np.array([0., 0., np.pi])
START_P = np.diag([0.01**2, 0.01**2, np.deg2rad(1.)**2])
# Only mission destinations are supplied; no intermediate route or wall coordinates.
GOALS = np.array([[-1.8,-1.2],[-4.8,-2.4],[-7.3,-2.4],[-7.8,0.]])
GOAL_NAMES = ['GOAL 1','GOAL 2','GOAL 3','Charger approach']
# Stand-off docking demonstration, not physical electrical contact.
# Marker-to-base separation .647 m; camera-to-marker approximately .557 m.
DOCK_POSE = np.array([-8.05, 0., np.pi])
CAMERA_MAX_AGE = 1.2

BAY_ESTIMATED_TOL = 0.03
APPROACH_TOL = 0.06
OVERSHOOT_TOL = 0.06
V_CRUISE = 0.16
OMEGA_MAX = 0.70
# Conservative circular envelope around base origin, plus margin.
ROBOT_RADIUS = 0.27
CLEARANCE_MARGIN = 0.05
LIDAR_OFFSET = np.array([-0.049, 0.])  # known sensor mounting from XML
MAX_RANGE = 12.0
CAMERA_USE_RANGE = 3.0  # provisional: farther camera was worse in prior tests

# Controlled encoder-increment noise experiment, NOT physical wheel slip.
# Independent zero-mean Gaussian travel errors: variance = coefficient^2 * travel.
# 0.005 m/sqrt(m) gives 5 mm cumulative wheel-distance SD over 1 m.
# This is a test severity, not a measured encoder specification. No noise at rest.
# Only the EKF input is perturbed; the low-level wheel servo keeps ideal encoders.
ENCODER_NOISE_ENABLED = True
ENCODER_NOISE_SEED = 41
ENCODER_NOISE_M_PER_SQRT_M = 0.005
USE_WHEEL_TRAVEL_Q = True

# The plate is a collision geom in physical_floor.xml. False gives the same
# insert normal floor traction for the control run. Set this once at launch.
SLIP_PLATE_ENABLED = False


class SlipPlate:
    """Score actual tyre-to-insert contacts for evaluation only."""
    def __init__(self, model, wheel_bodies, floor, plate_id):
        self.plate_id = plate_id
        if not SLIP_PLATE_ENABLED:
            model.geom_friction[plate_id] = model.geom_friction[floor]
        self.mu = float(model.geom_friction[plate_id, 0])
        self.geoms = []
        for body in wheel_bodies:
            ids = np.flatnonzero((model.geom_bodyid == body) &
                                 ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
            if len(ids) != 1:
                raise ValueError('Slip plate expects one collision tyre per wheel body.')
            self.geoms.append(int(ids[0]))
        self.on = np.zeros(2, dtype=bool)
        self.zone_time = 0.
        self.frames = 0
        self.travel = np.zeros(2)
        self.mismatch = np.zeros(2)
        self.contacts = np.zeros(2, dtype=int)
        self.mu_seen = set()

    def score(self, raw_changes, displacement, forward, data, dt):
        self.on[:] = False
        for j in range(data.ncon):
            contact = data.contact[j]
            if contact.efc_address < 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == self.plate_id else a if b == self.plate_id else -1
            if other in self.geoms:
                self.on[self.geoms.index(other)] = True
                self.mu_seen.add(round(float(contact.friction[0]), 5))
        if np.any(self.on):
            self.zone_time += dt
            wheel_motion = WHEEL_RADIUS * raw_changes
            hub_motion = displacement @ forward
            self.contacts += self.on
            self.travel[self.on] += abs(wheel_motion[self.on])
            self.mismatch[self.on] += abs(wheel_motion[self.on] - hub_motion[self.on])

    def summary(self):
        lines = [f'Slip plate enabled={SLIP_PLATE_ENABLED}; test sliding friction={self.mu}',
                 f'Time with actual tyre-plate contact: {self.zone_time:.2f} s',
                 f'Accepted camera updates during tyre-plate contact: {self.frames}',
                 f'Observed tyre-plate sliding friction: {sorted(self.mu_seen)}']
        for i,name in enumerate(('Left','Right')):
            pct = (f'{100*self.mismatch[i]/self.travel[i]:.2f}% of wheel travel'
                   if self.travel[i] > 1e-6 else 'insufficient wheel travel')
            lines.append(f'{name}: tyre-plate contact steps={self.contacts[i]}, '
                         f'wheel travel={self.travel[i]:.3f} m, accumulated absolute rolling-distance '
                         f'mismatch={100*self.mismatch[i]:.2f} cm ({pct}).')
        return lines


def noisy_wheel_increments(increments, rng):
    increments = np.asarray(increments, dtype=float)
    if not ENCODER_NOISE_ENABLED:
        return increments.copy()
    travel = WHEEL_RADIUS * increments
    sigma = ENCODER_NOISE_M_PER_SQRT_M * np.sqrt(np.abs(travel))
    return increments + rng.normal(0.0, sigma, size=2) / WHEEL_RADIUS


def lqr_gain(v):
    """Straight-path errors e=[left cross-track m, heading rad].
    e_y_dot=v*e_heading; e_heading_dot=omega. Exact ZOH at DT_CONTROL.
    Q normalises 10 cm and 15 deg error; R normalises 0.5 rad/s effort.
    This is tracking linearisation at nonzero forward speed, not at rest.
    """
    dt = DT_CONTROL
    A = np.array([[1., v*dt], [0., 1.]])
    B = np.array([[0.5*v*dt*dt], [dt]])
    Q = np.diag([1/0.10**2, 1/np.deg2rad(15.)**2])
    R = np.array([[1/0.5**2]])
    S = Q.copy()
    for _ in range(20000):
        K = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
        new = Q + A.T @ S @ A - A.T @ S @ B @ K
        if np.max(np.abs(new-S)) < 1e-10:
            S = new
            break
        S = new
    else:
        raise RuntimeError('DARE iteration did not converge.')
    K = np.linalg.solve(R+B.T@S@B, B.T@S@A)
    return K.ravel(), np.linalg.eigvals(A-B@K)


class Tracker:
    def __init__(self):
        self.segment = 0
        self.mode = 'ALIGN'
        self.route = np.zeros((2,2))
        # Gain scheduling for the nonzero speeds used near segment endpoints.
        self.speeds = np.array([0.025, 0.05, 0.075, 0.10, 0.13, 0.16])
        self.gains = [lqr_gain(v)[0] for v in self.speeds]

    def command(self, pose, endpoint_tol=0.02):
        if self.mode == 'DONE':
            return 0., 0., 0., 0.
        a, b = self.route[self.segment:self.segment+2]
        delta = b-a
        length = np.linalg.norm(delta)
        tangent = delta/length
        normal = np.array([-tangent[1], tangent[0]])
        remaining = length - np.dot(pose[:2]-a, tangent)
        ey = float(np.dot(pose[:2]-a, normal))
        eth = float(wrap_angle(pose[2]-np.arctan2(delta[1], delta[0])))
        distance = np.linalg.norm(pose[:2]-b)
        if distance <= (endpoint_tol if self.segment == len(self.route)-2 else 0.02):
            if self.segment == len(self.route)-2:
                self.mode = 'DONE'
            else:
                self.segment += 1
                self.mode = 'ALIGN'
            return 0., 0., ey, eth
        if remaining < -OVERSHOOT_TOL:
            self.mode = 'OVERSHOOT'
            return 0., 0., ey, eth
        if self.mode == 'ALIGN':
            if abs(eth) > np.deg2rad(4):
                return 0., float(np.clip(-1.5*eth, -OMEGA_MAX, OMEGA_MAX)), ey, eth
            self.mode = 'TRACK'
        # Large error leaves the local LQR regime: stop translation and realign.
        if abs(eth) > np.deg2rad(50):
            self.mode = 'ALIGN'
            return 0., 0., ey, eth
        v = float(np.clip(0.6*max(remaining, 0), 0.025, V_CRUISE))
        K = self.gains[int(np.argmin(abs(self.speeds-v)))]
        omega = float(np.clip(-K @ np.array([ey,eth]), -OMEGA_MAX, OMEGA_MAX))
        return v, omega, ey, eth


class Navigator:
    def __init__(self):
        self.tracker=Tracker()
        self.speeds=self.tracker.speeds
        self.planner=LidarPlanner()
        self.segment=0  # mission goal index, not temporary route segment
        self.mode='PLAN'
        self.have_path=False
        self.next_map=0.
        self.next_plan=0.
        self.plan_status='waiting'

    def command(self,pose,ranges,t):
        if self.mode=='DONE': return 0.,0.,0.,0.
        if t>=self.next_map:
            self.planner.update(pose,ranges)
            self.next_map=t+.25
        goal=GOALS[self.segment]
        arrival_tol = BAY_ESTIMATED_TOL if self.segment < len(GOALS)-1 else APPROACH_TOL
        if np.linalg.norm(pose[:2]-goal) <= arrival_tol:
            if self.segment==len(GOALS)-1:
                eth=float(wrap_angle(pose[2]-DOCK_POSE[2]))
                self.mode='FACE CHARGER'
                if abs(eth)>np.deg2rad(3):
                    return 0.,float(np.clip(-1.5*eth,-OMEGA_MAX,OMEGA_MAX)),0.,eth
                print('REACHED: Charger approach; starting visual docking')
                self.mode='DONE';return 0.,0.,0.,eth
            print('REACHED:',GOAL_NAMES[self.segment])
            self.segment+=1;goal=GOALS[self.segment];self.have_path=False
        if self.have_path:
            remaining=self.tracker.route[self.tracker.segment+1:]
            points=np.vstack([pose[:2],remaining])
            if not all(self.planner.line_clear(a,b) for a,b in zip(points[:-1],points[1:])):
                self.have_path=False
        if not self.have_path:
            self.mode='PLAN'
            if t<self.next_plan:return 0.,0.,0.,0.
            self.next_plan=t+1.
            path=self.planner.plan(pose,goal)
            self.plan_status=self.planner.status
            print(f'PLAN {GOAL_NAMES[self.segment]}: {self.plan_status}; '
                  f'points={0 if path is None else len(path)}')
            if path is None:return 0.,0.,0.,0.
            self.tracker.route=path;self.tracker.segment=0;self.tracker.mode='ALIGN'
            self.have_path=True
        bay_endpoint = self.segment < len(GOALS)-1 and np.linalg.norm(self.tracker.route[-1]-goal) < 1e-6
        command=self.tracker.command(pose, BAY_ESTIMATED_TOL if bay_endpoint else 0.02)
        self.mode=self.tracker.mode
        if self.mode in ('DONE','OVERSHOOT'):
            self.have_path=False;self.mode='PLAN'
            return 0.,0.,0.,0.
        return command


class Docking:
    """Fused-pose control, gated by fresh accepted camera observations.
    Visual confirmation uses a separately reconstructed camera pose, not truth.
    Polar pose feedback is used at low speed; route segments retain LQR.
    """
    def __init__(self):
        self.last_seen = -np.inf
        self.confirmations = 0
        self.visual_pose = None
        self.mode = 'DOCK ALIGN'

    def observe(self, z, t, fused_pose, active):
        self.last_seen = t
        r, bearing, heading = z
        direction = heading + bearing
        camera_xy = MARKER_XY - r*np.array([np.cos(direction), np.sin(direction)])
        base_xy = camera_xy - CAMERA_OFFSET*np.array([np.cos(heading), np.sin(heading)])
        self.visual_pose = np.array([*base_xy, heading])
        visual_ok = (np.linalg.norm(base_xy-DOCK_POSE[:2]) < .05
                     and abs(wrap_angle(heading-DOCK_POSE[2])) < np.deg2rad(3))
        fused_ok = (np.linalg.norm(fused_pose[:2]-DOCK_POSE[:2]) < .03
                    and abs(wrap_angle(fused_pose[2]-DOCK_POSE[2])) < np.deg2rad(3))
        self.confirmations = self.confirmations+1 if active and visual_ok and fused_ok else 0

    def command(self, pose, t):
        if t-self.last_seen > CAMERA_MAX_AGE:
            self.confirmations = 0
            self.mode = 'WAIT CAMERA'
            return 0., 0., 0., 0.
        delta = DOCK_POSE[:2]-pose[:2]
        distance = float(np.linalg.norm(delta))
        heading_error = float(wrap_angle(pose[2]-DOCK_POSE[2]))
        if self.confirmations >= 3:
            self.mode = 'DOCKED'
            return 0., 0., float(delta[1]), heading_error
        if distance < .02:
            self.mode = 'DOCK VERIFY'
            return 0., float(np.clip(-1.5*heading_error,-.25,.25)), float(delta[1]), heading_error
        alpha = float(wrap_angle(np.arctan2(delta[1],delta[0])-pose[2]))
        beta = float(wrap_angle(DOCK_POSE[2]-pose[2]-alpha))
        self.mode = 'DOCK APPROACH'
        v = min(.06,.5*distance) if abs(alpha)<np.deg2rad(35) else 0.
        omega = float(np.clip(1.4*alpha-.5*beta,-.30,.30))
        return v, omega, float(delta[1]), heading_error


def wheel_targets(v, omega):
    wheels = np.array([v-omega*WHEEL_SEPARATION/2,
                       v+omega*WHEEL_SEPARATION/2])/WHEEL_RADIUS
    return wheels * min(1., 5.5/max(np.max(abs(wheels)), 1e-12))


class WheelServo:
    """PI trim around XML velocity actuators, using measured encoder rate.
    Existing tests showed substantial target/actual speed deficit under load.
    Gains Kp=.5, Ki=2/s are provisional; bounded integral avoids windup.
    Outputs remain actuator velocity targets in rad/s, not torques.
    """
    def __init__(self):
        self.integral = np.zeros(2)

    def command(self, desired, measured, dt, stop=False):
        if stop:
            self.integral[:] = 0
            return np.zeros(2)
        error = desired-measured
        proposed = np.clip(self.integral+2.0*error*dt, -3., 3.)
        raw = desired+0.5*error+proposed
        saturated = np.clip(raw, -7.88, 7.88)
        keep = (raw == saturated) | ((raw-saturated)*error <= 0)
        self.integral[keep] = proposed[keep]
        return np.clip(desired+0.5*error+self.integral, -7.88, 7.88)


class Lidar:
    """Ground-truth transforms are confined to simulated sensor generation."""
    def __init__(self, model, base, site):
        self.model, self.base, self.site = model, base, site
        self.angles = np.deg2rad(np.arange(360))
        self.directions = np.column_stack([np.cos(self.angles), np.sin(self.angles), np.zeros(360)])
        self.robot_bodies = set()
        for body in range(model.nbody):
            parent = body
            while parent:
                if parent == base:
                    self.robot_bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.robot_geoms = np.array([g for g in range(model.ngeom)
                                    if int(model.geom_bodyid[g]) in self.robot_bodies], dtype=int)

    def read(self, data):
        import mujoco
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3,3)
        ranges = np.full(360, np.inf)
        hit = np.empty(1, dtype=np.int32)
        saved_groups = self.model.geom_group.copy()
        # Temporarily reserve group 5 for robot; restores before drawing.
        mask = np.ones(6, dtype=np.uint8)
        mask[5] = 0
        try:
            self.model.geom_group[:] = 0
            self.model.geom_group[self.robot_geoms] = 5
            for i, direction in enumerate(self.directions):
                d = mujoco.mj_ray(self.model, data, origin, rotation@direction,
                                  mask, True, -1, hit)
                if 0 <= d <= MAX_RANGE:
                    ranges[i] = d
        finally:
            self.model.geom_group[:] = saved_groups
        return ranges


def blocked_by_scan(ranges, v, omega):
    """Check sampled swept circular footprint over 0.8 s commanded motion.
    Finite 1-degree beams and assumed footprint/margins are not a safety proof.
    """
    valid = np.isfinite(ranges)
    if not np.any(valid):
        return True, np.nan  # fail closed on an empty scan in this enclosed room
    angles = np.deg2rad(np.arange(360))[valid]
    points = LIDAR_OFFSET + ranges[valid,None]*np.column_stack([np.cos(angles),np.sin(angles)])
    clearance = np.inf
    for t in np.linspace(0, 0.8, 17):
        if abs(omega) < 1e-8:
            center = np.array([v*t, 0.])
        else:
            center = np.array([v/omega*np.sin(omega*t), v/omega*(1-np.cos(omega*t))])
        clearance = min(clearance, float(np.min(np.linalg.norm(points-center,axis=1))-ROBOT_RADIUS))
    return clearance < CLEARANCE_MARGIN, clearance


def evaluation_pose(data, base):
    """Truth for reporting only. Never feed this into navigation or estimation."""
    rot = data.xmat[base].reshape(3, 3)
    return np.array([*data.xpos[base, :2], np.arctan2(rot[1, 0], rot[0, 0])])


def estimation_errors(estimate, truth):
    return (float(np.linalg.norm(estimate[:2]-truth[:2])),
            float(np.degrees(wrap_angle(estimate[2]-truth[2]))))


def write_records(path, fields, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    import mujoco
    import mujoco.viewer
    scene = ROOT / 'models/robotis_tb3/scene_assignment_plate.xml'
    if not scene.is_file():
        from runpy import run_path
        run_path(str(scene.with_name('install_physical_floor.py')))['install']()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    def required(kind, name):
        i = mujoco.mj_name2id(model,kind,name)
        if i < 0:
            raise RuntimeError('Missing model object: '+name)
        return i
    base = required(mujoco.mjtObj.mjOBJ_BODY,'base')
    floor = required(mujoco.mjtObj.mjOBJ_GEOM,'floor')
    wheel_bodies = [required(mujoco.mjtObj.mjOBJ_BODY,n) for n in ('wheel_left','wheel_right')]
    plate = SlipPlate(model,wheel_bodies,floor,required(mujoco.mjtObj.mjOBJ_GEOM,'slip_plate'))
    site = required(mujoco.mjtObj.mjOBJ_SITE,'lidar_site')
    joints = [required(mujoco.mjtObj.mjOBJ_JOINT,n) for n in ['wheel_left','wheel_right']]
    addresses = model.jnt_qposadr[joints]
    motors = [required(mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ['wheel_left','wheel_right']]
    # Retain original XML pose: no test_ekf.TEST_START or data.qpos placement.
    for _ in range(round(1/model.opt.timestep)):
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    run_label = 'encoder_noise' if ENCODER_NOISE_ENABLED else 'baseline'
    run_label += '_plate_on' if SLIP_PLATE_ENABLED else '_plate_off'
    run_label += '_wheel_Q' if USE_WHEEL_TRAVEL_Q else '_time_Q'
    run_label += f'_seed_{ENCODER_NOISE_SEED}'
    run_dir = ROOT / 'results' / ('navigation_' + run_label + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_dir.mkdir(parents=True, exist_ok=False)
    pose, P = START_ESTIMATE.copy(), START_P.copy()
    odom_pose = START_ESTIMATE.copy()  # evaluation-only shadow; same noisy increments
    goal_records, camera_records = [], []
    recorded_goals = set()

    def record_goal(name, target, t):
        if name in recorded_goals:
            return
        actual = evaluation_pose(data, base)
        pe, he = estimation_errors(pose, actual)
        oe, oh = estimation_errors(odom_pose, actual)
        actual_distance = float(np.linalg.norm(actual[:2]-target[:2]))
        goal_records.append(dict(goal=name, time_s=t,
            target_x=target[0], target_y=target[1],
            estimated_distance_m=float(np.linalg.norm(pose[:2]-target[:2])),
            actual_distance_m=actual_distance,
            fused_position_error_m=pe, fused_heading_error_deg=he,
            odom_position_error_m=oe, odom_heading_error_deg=oh,
            est_x=pose[0], est_y=pose[1], true_x=actual[0], true_y=actual[1]))
        recorded_goals.add(name)
    previous = data.qpos[addresses].copy()
    encoder_rng = np.random.default_rng(ENCODER_NOISE_SEED)
    accumulated_noise = np.zeros(2)
    control_encoders = previous.copy()
    tracker, servo = Navigator(), WheelServo()
    docking = Docking()
    docking_started = None
    lidar = Lidar(model,base,site)
    camera = MarkerCamera(model)
    next_control=next_camera=next_print=0.
    last_control=data.time
    start=data.time
    status='START'
    camera_status='waiting'
    accepted=detected=far=0
    blocked_since=done_since=None
    v=omega=ey=eth=0.
    clearance=np.nan
    rows=[]
    outcome='Viewer closed'
    print('Output folder:', run_dir)
    print('ALL 3 GOALS + VISUAL DOCK: EKF feedback, LQR route, LiDAR stop')
    print(f'Encoder noise: {ENCODER_NOISE_ENABLED}; seed={ENCODER_NOISE_SEED}; '
          f'coefficient={ENCODER_NOISE_M_PER_SQRT_M} m/sqrt(m).')
    print('Encoder noise affects measured increments. Slip plate changes simulated tyre traction.')
    print('Mission goals:',GOALS.tolist())
    print('LiDAR map + shortest-path/frontier planning; no supplied obstacle map.')
    print('Declared initial pose:',START_ESTIMATE)
    print(f'Camera corrections limited to measured range <= {CAMERA_USE_RANGE:g} m.')
    print('At the original start the marker may be invisible; prediction continues.')
    for speed in tracker.speeds:
        gain,eigs=lqr_gain(speed)
        print(f'LQR v={speed:.3f}: K={gain}; closed-loop poles={eigs}')
    try:
        with mujoco.viewer.launch_passive(model,data) as viewer:
            while viewer.is_running():
                wall=time.perf_counter()
                with viewer.lock():
                    hubs_before = data.xpos[wheel_bodies,:2].copy()
                    forward = data.xmat[base].reshape(3,3)[:2,0].copy()
                    forward /= max(np.linalg.norm(forward),1e-12)
                    mujoco.mj_step(model,data)
                    mujoco.mj_forward(model,data)
                    enc=data.qpos[addresses].copy()
                    raw_increments=enc-previous
                    plate.score(raw_increments, data.xpos[wheel_bodies,:2]-hubs_before, forward, data, model.opt.timestep)
                    measured_increments=noisy_wheel_increments(raw_increments,encoder_rng)
                    accumulated_noise += WHEEL_RADIUS*(measured_increments-raw_increments)
                    pose, P = predict_ekf(
                        pose,
                        P,
                        measured_increments,
                        model.opt.timestep,
                        wheel_noise_coeff=(
                            ENCODER_NOISE_M_PER_SQRT_M if USE_WHEEL_TRAVEL_Q else None
                        ),
                    )   
                    odom_pose,_=motion(odom_pose,measured_increments)
                    previous=enc
                    t=data.time-start
                    if t>=next_camera:
                        next_camera=t+DT_CAMERA
                        z=camera.read(data)
                        camera_status=camera.status
                        if z is None:
                            docking.confirmations=0
                        if z is not None:
                            detected+=1
                            if z[0] <= CAMERA_USE_RANGE:
                                # Both errors use identical simulation state: no motion between.
                                camera_truth=evaluation_pose(data,base)
                                before_pos,before_yaw=estimation_errors(pose,camera_truth)
                                pose,P,nis,ok,_=correct_ekf(pose,P,z)
                                after_pos,after_yaw=estimation_errors(pose,camera_truth)
                                camera_records.append(dict(time_s=t, accepted=int(ok), nis=nis,
                                    range_m=z[0], before_position_error_m=before_pos,
                                    after_position_error_m=after_pos,
                                    position_improvement_m=before_pos-after_pos,
                                    before_heading_error_deg=before_yaw,
                                    after_heading_error_deg=after_yaw,
                                    heading_improvement_deg=abs(before_yaw)-abs(after_yaw)))
                                plate.frames += int(ok and np.any(plate.on))
                                accepted+=int(ok)
                                if ok:
                                    docking.observe(z,t,pose,tracker.mode=='DONE')
                                else:
                                    docking.confirmations=0
                                camera_status='accepted' if ok else 'innovation rejected'
                            else:
                                far+=1
                                docking.confirmations=0
                                camera_status='too far: prediction only'
                    if t>=next_control:
                        elapsed=data.time-last_control
                        rates=(enc-control_encoders)/elapsed
                        control_encoders=enc.copy()
                        last_control=data.time
                        next_control=t+DT_CONTROL
                        ranges=lidar.read(data)
                        previous_goal=tracker.segment
                        if tracker.mode != 'DONE':
                            v,omega,ey,eth=tracker.command(pose,ranges,t)
                        if tracker.segment != previous_goal:
                            record_goal(GOAL_NAMES[previous_goal],GOALS[previous_goal],t)
                        if tracker.mode == 'DONE':
                            record_goal(GOAL_NAMES[-1],GOALS[-1],t)
                        if tracker.mode == 'DONE':
                            docking_started = t if docking_started is None else docking_started
                            v,omega,ey,eth=docking.command(pose,t)
                        blocked,clearance=blocked_by_scan(ranges,v,omega)
                        status=docking.mode if tracker.mode=='DONE' else tracker.mode
                        stop=status in ('DOCKED','WAIT CAMERA','OVERSHOOT','PLAN')
                        if blocked and not stop:
                            status='LIDAR STOP'
                            blocked_since=t if blocked_since is None else blocked_since
                            stop=True
                            if tracker.mode!='DONE':
                                tracker.have_path=False
                        else:
                            blocked_since=None
                        data.ctrl[motors]=servo.command(wheel_targets(v,omega),rates,elapsed,stop)
                        # Evaluation-only ground truth below, never passed to tracker/EKF.
                        truth=evaluation_pose(data,base)
                        if status=='DOCKED':
                            record_goal('Charger dock',DOCK_POSE,t)
                        rows.append([t,*pose,*truth,*np.diag(P),tracker.segment,ey,eth,v,omega,
                                     *data.ctrl[motors],clearance,status,camera_status,
                                     *accumulated_noise,int(ENCODER_NOISE_ENABLED),
                                     ENCODER_NOISE_SEED,ENCODER_NOISE_M_PER_SQRT_M,int(plate.on[0]),int(plate.on[1]),*plate.mismatch,*odom_pose])
                        if t>=next_print:
                            next_print=t+1
                            print(f'{t:6.1f}s seg={tracker.segment+1} {status:12s} '
                                  f'est=({pose[0]:+.3f},{pose[1]:+.3f}) '
                                  f'ey={ey:+.3f}m heading_err={np.degrees(eth):+.1f}deg '
                                  f'clearance={clearance:.2f}m camera={camera_status}')
                        if status=='DOCKED':
                            done_since=t if done_since is None else done_since
                            if t-done_since>=2:
                                outcome='ALL GOALS REACHED + CAMERA-CONFIRMED STAND-OFF DOCK'
                                break
                        if status != 'DOCKED':
                            done_since=None
                        if docking_started is not None and t-docking_started>60:
                            outcome='Docking timeout: visual confirmation or alignment not achieved'
                            break
                        if tracker.mode=='OVERSHOOT':
                            outcome='Stopped: segment overshoot; inspect tracking log'
                            break
                        if blocked_since is not None and t-blocked_since>15:
                            outcome='Stopped: LiDAR blockage persisted 15 seconds despite replanning'
                            break
                    if t>=MAX_TIME:
                        outcome='Timeout: goal not reached'
                        break
                viewer.sync()
                delay=model.opt.timestep-(time.perf_counter()-wall)
                if delay>0:
                    time.sleep(delay)
    finally:
        data.ctrl[motors]=0
        camera.close()
    print('\nNAVIGATION RESULT:',outcome)
    if rows:
        print('Final estimated pose:',pose)
        print('Final actual pose (evaluation only):',truth)
        print(f'Estimated goal distance: {np.linalg.norm(pose[:2]-DOCK_POSE[:2]):.4f} m')
        print(f'Actual goal distance: {np.linalg.norm(truth[:2]-DOCK_POSE[:2]):.4f} m')
        print(f'Final localisation error: {100*np.linalg.norm(pose[:2]-truth[:2]):.2f} cm')
        samples=np.asarray([row[1:7] for row in rows],dtype=float)
        position_errors=np.linalg.norm(samples[:,:2]-samples[:,3:5],axis=1)
        yaw_errors=wrap_angle(samples[:,2]-samples[:,5])
        print(f'Localisation position RMSE: {100*np.sqrt(np.mean(position_errors**2)):.2f} cm; '
              f'max: {100*np.max(position_errors):.2f} cm')
        print(f'Localisation heading RMSE: {np.degrees(np.sqrt(np.mean(yaw_errors**2))):.3f} deg')
        print(f'Camera detected={detected}, accepted={accepted}, skipped_far={far}')
        output=run_dir/'trajectory.csv'
        with output.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['time','est_x','est_y','est_heading','true_x','true_y','true_heading',
                        'P_xx','P_yy','P_heading','segment','cross_track_m','heading_error_rad',
                        'requested_v','requested_omega','left_command','right_command',
                        'predicted_clearance_m','status','camera_status',
                        'left_accumulated_encoder_noise_m','right_accumulated_encoder_noise_m',
                        'encoder_noise_enabled','encoder_noise_seed','encoder_noise_m_per_sqrt_m',
                        'left_plate_contact','right_plate_contact','left_cumulative_mismatch_m',
                        'right_cumulative_mismatch_m','odom_x','odom_y','odom_heading'])
            w.writerows(rows)
        print('Log:',output)
        map_path=run_dir/'lidar_map.npz'
        tracker.planner.save(map_path)
        print('LiDAR map:',map_path)
        # Same-run estimator comparison; the shadow never controls the robot.
        odom_samples=np.asarray([row[-3:] for row in rows],dtype=float)
        odom_errors=np.linalg.norm(odom_samples[:,:2]-samples[:,3:5],axis=1)
        odom_yaws=wrap_angle(odom_samples[:,2]-samples[:,5])
        comparison=[]
        for name,errors,yaws in [('Fused EKF',position_errors,yaw_errors),
                                 ('Odometry only',odom_errors,odom_yaws)]:
            comparison.append(dict(estimator=name,
                position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                max_position_error_m=float(np.max(errors)),
                final_position_error_m=float(errors[-1]),
                heading_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaws**2))))))
        write_records(run_dir/'estimator_comparison.csv',list(comparison[0]),comparison)
        goal_fields = [
            'goal', 'time_s', 'target_x', 'target_y',
            'estimated_distance_m', 'actual_distance_m',
            'fused_position_error_m', 'fused_heading_error_deg',
            'odom_position_error_m', 'odom_heading_error_deg',
            'est_x', 'est_y', 'true_x', 'true_y',
        ]
        write_records(run_dir/'goal_errors.csv',goal_fields,goal_records)
        camera_fields=['time_s','accepted','nis','range_m','before_position_error_m',
            'after_position_error_m','position_improvement_m','before_heading_error_deg',
            'after_heading_error_deg','heading_improvement_deg']
        write_records(run_dir/'camera_corrections.csv',camera_fields,camera_records)
        summary=[outcome, f'Encoder noise enabled={ENCODER_NOISE_ENABLED}; seed={ENCODER_NOISE_SEED}; coefficient={ENCODER_NOISE_M_PER_SQRT_M}',
            'GOAL ERRORS AT DECLARED ARRIVAL (cm; truth for evaluation only)',
            'Goal                  Estimated distance  Actual distance  Localisation error']
        for name in [*GOAL_NAMES,'Charger dock']:
            record=next((r for r in goal_records if r['goal']==name),None)
            if record is None:
                summary.append(f'{name:22s} NOT REACHED')
            else:
                summary.append(f"{name:22s} {100*record['estimated_distance_m']:18.2f} {100*record['actual_distance_m']:16.2f} {100*record['fused_position_error_m']:19.2f}")
                reached_bays = sum(
            name in recorded_goals for name in GOAL_NAMES[:-1]
        )
        summary.append(
            f"Bay arrivals declared: {reached_bays}/{len(GOAL_NAMES)-1} "
            f"using a {100*BAY_ESTIMATED_TOL:g} cm estimated-position threshold."
        )
        summary.append(
            "Actual distance and localisation error at each declared arrival "
            "are reported above for accuracy evaluation."
        )
        summary += ['', 'SAME-RUN ESTIMATOR COMPARISON',
                    'Estimator         Position RMSE(cm)  Max(cm)  Final(cm)  Heading RMSE(deg)']
        for r in comparison:
            summary.append(f"{r['estimator']:17s} {100*r['position_rmse_m']:17.2f} {100*r['max_position_error_m']:8.2f} {100*r['final_position_error_m']:10.2f} {r['heading_rmse_deg']:18.3f}")
        updates=[r for r in camera_records if r['accepted']]
        if updates:
            gains=np.array([r['position_improvement_m'] for r in updates])
            summary += [f'Accepted camera updates: {len(updates)}; position improved: {sum(gains>1e-9)}; worsened: {sum(gains < -1e-9)}; unchanged: {sum(abs(gains)<=1e-9)}',
                f'Mean immediate position-error reduction: {100*gains.mean():+.3f} cm (positive is improvement).']
        else:
            summary.append('No accepted camera updates: this run cannot demonstrate camera benefit.')
        summary += ['Odometry shadow uses the same noisy increments and never controls movement.',
                    'Goal rows are sampled at first declared arrival; final docking metrics may differ after settling.',
                    'This comparison tests estimation on one trajectory, not an odometry-only controlled mission.',
                    'No collision detection metric is included.',
                    'Camera SD settings: range=0.02 m, bearing=0.5 deg, heading=5.0 deg.',]
        summary += ['', *plate.summary()]
        report='\n'.join(summary)+'\n'
        (run_dir/'summary.txt').write_text(report,encoding='utf-8')
        print('\n'+report)
        print('Docking target:',DOCK_POSE)
        print('Latest camera-derived pose:',docking.visual_pose)
        print('Consecutive docking confirmations:',docking.confirmations)

if __name__=='__main__':
    main()
