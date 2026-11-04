"""Track A milestone: estimated-state LQR tracking with LiDAR stop.
Place beside the supplied orientation-aware test_ekf.py. Its main() is NOT run.
Run: python navigate_goal1.py
Route is predefined, not an obstacle planner. No XML changes or teleportation.
All dynamics/controller parameters are provisional, documented below.
"""
from pathlib import Path
import csv
import time
import numpy as np
from test_ekf import (MarkerCamera, predict_ekf, correct_ekf,
                      wrap_angle, WHEEL_RADIUS, WHEEL_SEPARATION)

ROOT = Path(__file__).resolve().parent
DT_CONTROL = 0.05
DT_CAMERA = 0.5
MAX_TIME = 120.0
# Declared nominal launch pose from supplied XML, NOT read from simulation truth.
START_ESTIMATE = np.array([0., 0., np.pi])
START_P = np.diag([0.01**2, 0.01**2, np.deg2rad(1.)**2])
ROUTE = np.array([[0., 0.], [-1.8, 0.], [-1.8, -1.2]])
GOAL_TOL = 0.06
V_CRUISE = 0.10
OMEGA_MAX = 0.55
# Conservative circular envelope around base origin, plus margin.
ROBOT_RADIUS = 0.27
CLEARANCE_MARGIN = 0.05
LIDAR_OFFSET = np.array([-0.049, 0.])  # known sensor mounting from XML
MAX_RANGE = 12.0
CAMERA_USE_RANGE = 3.0  # provisional: farther camera was worse in prior tests


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
        # Gain scheduling for the nonzero speeds used near segment endpoints.
        self.speeds = np.array([0.025, 0.05, 0.075, 0.10])
        self.gains = [lqr_gain(v)[0] for v in self.speeds]

    def command(self, pose):
        if self.mode == 'DONE':
            return 0., 0., 0., 0.
        a, b = ROUTE[self.segment:self.segment+2]
        delta = b-a
        length = np.linalg.norm(delta)
        tangent = delta/length
        normal = np.array([-tangent[1], tangent[0]])
        remaining = length - np.dot(pose[:2]-a, tangent)
        ey = float(np.dot(pose[:2]-a, normal))
        eth = float(wrap_angle(pose[2]-np.arctan2(delta[1], delta[0])))
        distance = np.linalg.norm(pose[:2]-b)
        if distance < GOAL_TOL:
            if self.segment == len(ROUTE)-2:
                self.mode = 'DONE'
            else:
                self.segment += 1
                self.mode = 'ALIGN'
            return 0., 0., ey, eth
        if remaining < -GOAL_TOL:
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


def main():
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(str(ROOT/'models/robotis_tb3/scene_assignment.xml'))
    data = mujoco.MjData(model)
    def required(kind, name):
        i = mujoco.mj_name2id(model,kind,name)
        if i < 0:
            raise RuntimeError('Missing model object: '+name)
        return i
    base = required(mujoco.mjtObj.mjOBJ_BODY,'base')
    site = required(mujoco.mjtObj.mjOBJ_SITE,'lidar_site')
    joints = [required(mujoco.mjtObj.mjOBJ_JOINT,n) for n in ['wheel_left','wheel_right']]
    addresses = model.jnt_qposadr[joints]
    motors = [required(mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in ['wheel_left','wheel_right']]
    # Retain original XML pose: no test_ekf.TEST_START or data.qpos placement.
    for _ in range(round(1/model.opt.timestep)):
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    pose, P = START_ESTIMATE.copy(), START_P.copy()
    previous = data.qpos[addresses].copy()
    control_encoders = previous.copy()
    tracker, servo = Tracker(), WheelServo()
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
    print('GOAL 1: XML start, EKF feedback, LQR straight segments, LiDAR stop')
    print('Route:',ROUTE.tolist())
    print('Declared initial pose:',START_ESTIMATE)
    print('Camera corrections limited to measured range <= 3m (provisional).')
    print('At the original start the marker may be invisible; prediction continues.')
    for speed in tracker.speeds:
        gain,eigs=lqr_gain(speed)
        print(f'LQR v={speed:.3f}: K={gain}; closed-loop poles={eigs}')
    try:
        with mujoco.viewer.launch_passive(model,data) as viewer:
            while viewer.is_running():
                wall=time.perf_counter()
                with viewer.lock():
                    mujoco.mj_step(model,data)
                    mujoco.mj_forward(model,data)
                    enc=data.qpos[addresses].copy()
                    pose,P=predict_ekf(pose,P,enc-previous,model.opt.timestep)
                    previous=enc
                    t=data.time-start
                    if t>=next_camera:
                        next_camera=t+DT_CAMERA
                        z=camera.read(data)
                        camera_status=camera.status
                        if z is not None:
                            detected+=1
                            if z[0] <= CAMERA_USE_RANGE:
                                pose,P,nis,ok,_=correct_ekf(pose,P,z)
                                accepted+=int(ok)
                                camera_status='accepted' if ok else 'innovation rejected'
                            else:
                                far+=1
                                camera_status='too far: prediction only'
                    if t>=next_control:
                        elapsed=data.time-last_control
                        rates=(enc-control_encoders)/elapsed
                        control_encoders=enc.copy()
                        last_control=data.time
                        next_control=t+DT_CONTROL
                        v,omega,ey,eth=tracker.command(pose)
                        ranges=lidar.read(data)
                        blocked,clearance=blocked_by_scan(ranges,v,omega)
                        status=tracker.mode
                        stop=tracker.mode in ('DONE','OVERSHOOT')
                        if blocked and not stop:
                            status='LIDAR STOP'
                            blocked_since=t if blocked_since is None else blocked_since
                            stop=True
                        else:
                            blocked_since=None
                        data.ctrl[motors]=servo.command(wheel_targets(v,omega),rates,elapsed,stop)
                        # Evaluation-only ground truth below, never passed to tracker/EKF.
                        rot=data.xmat[base].reshape(3,3)
                        truth=np.array([*data.xpos[base,:2],np.arctan2(rot[1,0],rot[0,0])])
                        rows.append([t,*pose,*truth,*np.diag(P),tracker.segment,ey,eth,v,omega,
                                     *data.ctrl[motors],clearance,status,camera_status])
                        if t>=next_print:
                            next_print=t+1
                            print(f'{t:6.1f}s seg={tracker.segment+1} {status:12s} '
                                  f'est=({pose[0]:+.3f},{pose[1]:+.3f}) '
                                  f'ey={ey:+.3f}m heading_err={np.degrees(eth):+.1f}deg '
                                  f'clearance={clearance:.2f}m camera={camera_status}')
                        if tracker.mode=='DONE':
                            done_since=t if done_since is None else done_since
                            if t-done_since>=2:
                                outcome='GOAL REACHED (estimated position within 6 cm)'
                                break
                        if tracker.mode=='OVERSHOOT':
                            outcome='Stopped: segment overshoot; inspect tracking log'
                            break
                        if blocked_since is not None and t-blocked_since>5:
                            outcome='Stopped: LiDAR blockage persisted 5 seconds; no automatic rerouting yet'
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
        print(f'Estimated goal distance: {np.linalg.norm(pose[:2]-ROUTE[-1]):.4f} m')
        print(f'Actual goal distance: {np.linalg.norm(truth[:2]-ROUTE[-1]):.4f} m')
        print(f'Final localisation error: {100*np.linalg.norm(pose[:2]-truth[:2]):.2f} cm')
        print(f'Camera detected={detected}, accepted={accepted}, skipped_far={far}')
        output=ROOT/('navigation_goal1_'+time.strftime('%Y%m%d_%H%M%S')+'.csv')
        with output.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['time','est_x','est_y','est_heading','true_x','true_y','true_heading',
                        'P_xx','P_yy','P_heading','segment','cross_track_m','heading_error_rad',
                        'requested_v','requested_omega','left_command','right_command',
                        'predicted_clearance_m','status','camera_status'])
            w.writerows(rows)
        print('Log:',output.name)

if __name__=='__main__':
    main()
