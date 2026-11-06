"""Wheel + ArUco EKF integration test. Run beside test_camera.py.
Requires existing models/ and calibration/camera_calibration.npz.
No simulated body/camera pose is supplied to the estimator.
Q and R below are provisional assumptions, not measured sensor statistics.
The camera model assumes planar motion (negligible pitch/roll).
Uses the known marker front-normal direction plus measured PnP rotation.
Planar PnP orientation can be noisy/ambiguous; inspect camera heading diagnostics.
"""
from pathlib import Path
import time
from datetime import datetime
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.288
CAMERA_OFFSET = 0.09
# Known map: visible front surface, 1 mm ahead of panel centre.
MARKER_XY = np.array([-8.697, 0.0])
MARKER_NORMAL_YAW = 0.0  # XML panel front normal points along world +x.
MARKER_SIZE = 0.36
WIDTH, HEIGHT = 2560, 1920
CAMERA_PERIOD = 0.2  # simulation seconds; high-res rendering may run slowly
# Physical placement is a test fixture, not a measurement supplied to the filter.
TEST_START = np.array([-3.0, 0.0, np.deg2rad(180.0)])
# Explicit declared prior. Keep the previous deliberate offset to compare tests.
# For an accurate-prior comparison, change only INITIAL_ESTIMATE to [-6, 0, pi].
INITIAL_ESTIMATE = np.array([-3.00, 0.0, np.deg2rad(180.0)])
INITIAL_P = np.diag([0.15**2, 0.15**2, np.deg2rad(5.0)**2])
# Provisional standard deviations: 2 cm range, 0.5 deg bearing, 5 deg heading.
# All three come from the same corners: diagonal R ignores their correlation.
# This is an integration test, not a calibrated covariance/consistency claim.
# Experimental standard deviations: 5 mm, 0.05 deg, 1 deg.
# Inflated above static-test errors to allow for motion and model mismatch.
R_CAMERA = np.diag([
    0.02**2,
    np.deg2rad(0.5)**2,
    np.deg2rad(5.0)**2,
])
# Provisional continuous covariance rate; multiplied by elapsed seconds.
Q_RATE = np.diag([0.002**2, 0.002**2, np.deg2rad(0.2)**2])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(pose, wheel_changes):
    dl, dr = WHEEL_RADIUS * np.asarray(wheel_changes)
    ds = (dl + dr) / 2
    dtheta = (dr - dl) / WHEEL_SEPARATION
    step = ds * np.sinc(dtheta / (2 * np.pi))
    direction = pose[2] + dtheta / 2
    dx, dy = step * np.cos(direction), step * np.sin(direction)
    result = pose + np.array([dx, dy, dtheta])
    result[2] = wrap_angle(result[2])
    F = np.array([[1., 0., -dy], [0., 1., dx], [0., 0., 1.]])
    return result, F


def predict_ekf(pose, P, wheel_changes, dt, wheel_noise_coeff=None):
    result, F = motion(pose, wheel_changes)

    if wheel_noise_coeff is None:
        # Original model: covariance added per second.
        Q = Q_RATE * dt
    else:
        # Wheel changes arrive in radians. Convert to metres.
        dl, dr = WHEEL_RADIUS * np.asarray(wheel_changes, dtype=float)

        # The simulator makes each wheel's error variance proportional
        # to its travel. The EKF only has measured travel available.
        wheel_variances = wheel_noise_coeff**2 * np.abs([dl, dr])

        # First-order approximation over one small physics step.
        # Each column describes how one wheel's travel error changes
        # the robot's estimated x, y and heading.
        mid_heading = pose[2] + (dr - dl) / (2 * WHEEL_SEPARATION)
        c = np.cos(mid_heading)
        s = np.sin(mid_heading)
        G = np.array([
            [0.5 * c,               0.5 * c],
            [0.5 * s,               0.5 * s],
            [-1 / WHEEL_SEPARATION, 1 / WHEEL_SEPARATION],
        ])

        Q = G @ np.diag(wheel_variances) @ G.T

    P = F @ P @ F.T + Q
    return result, (P + P.T) / 2


def camera_model(pose):
    """Expected range, bearing, world heading; angles in radians."""
    theta = pose[2]
    c, s = np.cos(theta), np.sin(theta)
    camera_xy = pose[:2] + CAMERA_OFFSET * np.array([c, s])
    dx, dy = MARKER_XY - camera_xy
    q = dx * dx + dy * dy
    if q < 1e-12:
        raise ValueError('Camera too close to landmark for range/bearing model.')
    r = np.sqrt(q)
    z = np.array([r, wrap_angle(np.arctan2(dy, dx) - theta), wrap_angle(theta)])
    H = np.array([
        [-dx/r, -dy/r, CAMERA_OFFSET * (dx*s - dy*c)/r],
        [dy/q, -dx/q, -1 - CAMERA_OFFSET * (dx*c + dy*s)/q],
        [0., 0., 1.],
    ])
    return z, H


def correct_ekf(pose, P, z):
    expected, H = camera_model(pose)
    innovation = z - expected
    innovation[1:] = wrap_angle(innovation[1:])
    S = H @ P @ H.T + R_CAMERA
    nis = float(innovation @ np.linalg.solve(S, innovation))
    # Broad diagnostic gate, not a calibrated acceptance guarantee.
    if not np.isfinite(nis) or nis > 25.0:
        return pose, P, nis, False, innovation
    K = np.linalg.solve(S, H @ P).T
    result = pose + K @ innovation
    result[2] = wrap_angle(result[2])
    A = np.eye(3) - K @ H
    P = A @ P @ A.T + K @ R_CAMERA @ K.T  # Joseph form
    return result, (P + P.T) / 2, nis, True, innovation


def heading_from_normal(normal):
    """OpenCV normal (right/down/forward) -> robot heading in world.

    Camera faces robot +x; camera right is robot -y. Thus the normal in
    robot horizontal coordinates is [normal_z, -normal_x]. Its world
    direction is known from the marker map. No simulator pose is used.
    The object-point ordering makes +z the printed marker front normal.
    """
    nx, ny, nz = normal
    if np.hypot(nx, nz) < 1e-6:
        raise ValueError('Marker normal has no usable horizontal component.')
    return wrap_angle(MARKER_NORMAL_YAW - np.arctan2(-nx, nz))


class MarkerCamera:
    def __init__(self, model):
        import cv2
        import mujoco
        self.cv2 = cv2
        self.status = 'waiting'
        with np.load(ROOT / 'calibration/camera_calibration.npz') as calibration:
            self.K = calibration['camera_matrix'].astype(float).copy()
            self.distortion = calibration['distortion'].copy()
            cw, ch = map(int, calibration['image_size'])
        sx, sy = WIDTH/cw, HEIGHT/ch
        if not np.isclose(sx, sy):
            raise ValueError('Calibration and rendering must have matching aspect ratios.')
        self.K[0, 0] *= sx
        self.K[1, 1] *= sy
        self.K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        self.K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 8
        params.relativeCornerRefinmentWinSize = 0.5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.001
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        h = MARKER_SIZE / 2
        self.object_points = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]], dtype=float)
        model.vis.global_.offwidth = WIDTH
        model.vis.global_.offheight = HEIGHT
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    def read(self, data):
        """Pixels -> PnP -> range, bearing and world heading. No truth readback."""
        cv2 = self.cv2
        self.status = 'not detected'
        self.renderer.update_scene(data, camera='front_camera')
        rgb = self.renderer.render()
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return None
        matches = np.flatnonzero(ids.flatten() == 0)
        if len(matches) != 1:
            return None
        points = corners[matches[0]].reshape(4, 2).astype(float)
        ok, rvec, tvec = cv2.solvePnP(self.object_points, points, self.K,
                                    self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec is None or rvec is None:
            return None
        if not np.all(np.isfinite(tvec)) or not np.all(np.isfinite(rvec)) or tvec[2, 0] <= 0:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        normal = rotation[:, 2]
        # Reject a back-facing solution, rather than silently flipping its normal.
        if np.dot(normal, -tvec.ravel()) <= 0:
            self.status = 'PnP rejected: back-facing normal'
            return None
        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec,
                                          self.K, self.distortion)
        reprojection = np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - points)**2, axis=1)))
        if not np.isfinite(reprojection) or reprojection > 2.0:
            self.status = 'PnP rejected: reprojection error'
            return None
        # A low reprojection error does not guarantee accurate planar orientation.
        camera_heading = heading_from_normal(normal)
        xr, _, zf = tvec.ravel()
        self.status = f'PnP reprojection={reprojection:.3f}px'
        return np.array([np.hypot(xr, zf), np.arctan2(-xr, zf), camera_heading])

    def close(self):
        self.renderer.close()


def main():
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'models/robotis_tb3/scene_assignment.xml'))
    data = mujoco.MjData(model)

    def required_id(kind, name):
        result = mujoco.mj_name2id(model, kind, name)
        if result < 0:
            raise ValueError(f'Missing model object: {name}')
        return result

    wheel_joints = [required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ('wheel_left', 'wheel_right')]
    wheel_addresses = model.jnt_qposadr[wheel_joints]
    motors = [required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ('wheel_left', 'wheel_right')]
    base = required_id(mujoco.mjtObj.mjOBJ_BODY, 'base')
    free_joint = required_id(mujoco.mjtObj.mjOBJ_JOINT, 'base_joint')
    address = model.jnt_qposadr[free_joint]
    # Test fixture only: move robot to previously used camera test location.
    data.qpos[address:address+2] = TEST_START[:2]
    half_heading = TEST_START[2] / 2
    data.qpos[address+3:address+7] = [np.cos(half_heading), 0., 0., np.sin(half_heading)]
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    def evaluation_pose():
        rot = data.xmat[base].reshape(3, 3)
        return np.array([*data.xpos[base, :2], np.arctan2(rot[1, 0], rot[0, 0])])

    odom = INITIAL_ESTIMATE.copy()
    ekf = INITIAL_ESTIMATE.copy()
    P = INITIAL_P.copy()
    previous = data.qpos[wheel_addresses].copy()
    camera = MarkerCamera(model)
    rows = []
    camera_heading_errors = []
    counts = {'attempts': 0, 'detected': 0, 'accepted': 0, 'rejected': 0, 'disabled': 0}
    start = data.time
    next_camera = 0.0
    next_print = 0.0
    initial_truth = evaluation_pose()  # evaluation only
    print('ORIENTATION-AWARE EKF: range + bearing + marker-derived heading')
    print('Running file:', Path(__file__).resolve())
    print('Physical test start [x, y, deg]:', *TEST_START[:2], np.degrees(TEST_START[2]))
    print('Estimator prior [x, y, deg]:', *INITIAL_ESTIMATE[:2], np.degrees(INITIAL_ESTIMATE[2]))
    print('Deliberate prior offset [dx, dy, deg]:',
          *(INITIAL_ESTIMATE[:2]-TEST_START[:2]),
          np.degrees(wrap_angle(INITIAL_ESTIMATE[2]-TEST_START[2])))
    print('Q/R are provisional. No deliberate sensor noise is added.')
    print('Camera corrections intentionally disabled from 3 to 5 seconds.')
    print('Time   Odom error(cm)  EKF error(cm)  EKF yaw error(deg)  Camera')
    status = 'waiting'
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time - start < 8.0:
                wall_start = time.perf_counter()
                t = data.time - start
                # Short forward drive, gentle left/right arcs, then stop.
                command = (4., 4.) if t < 3 else ((3., 5.) if t < 4 else ((5., 3.) if t < 5 else ((4., 4.) if t < 6 else (0., 0.))))
                data.ctrl[motors] = command
                mujoco.mj_step(model, data)
                # Refresh transforms to the same state as the wheel positions.
                mujoco.mj_forward(model, data)
                encoders = data.qpos[wheel_addresses].copy()
                increments = encoders - previous
                previous = encoders
                odom, _ = motion(odom, increments)
                ekf, P = predict_ekf(ekf, P, increments, model.opt.timestep)
                t = data.time - start
                z = None
                nis = np.nan
                accepted = False
                innovation = np.full(3, np.nan)
                if t >= next_camera:
                    next_camera += CAMERA_PERIOD
                    if 3.0 <= t < 5.0:
                        counts['disabled'] += 1
                        status = 'disabled for dropout test'
                    else:
                        counts['attempts'] += 1
                        z = camera.read(data)
                        status = camera.status
                        if z is not None:
                            counts['detected'] += 1
                            ekf, P, nis, accepted, innovation = correct_ekf(ekf, P, z)
                            counts['accepted' if accepted else 'rejected'] += 1
                            status = f'{"accepted" if accepted else "rejected"}, NIS={nis:.2f}'
                truth = evaluation_pose()
                if z is not None:
                    camera_heading_error = np.degrees(wrap_angle(z[2] - truth[2]))
                    camera_heading_errors.append(camera_heading_error)
                    # Evaluation only: never feeds back to camera or filter.
                    if len(camera_heading_errors) <= 5:
                        print(f'  CAMERA CHECK: heading={np.degrees(z[2]):+.3f} deg, '
                              f'truth={np.degrees(truth[2]):+.3f} deg, '
                              f'error={camera_heading_error:+.3f} deg; {camera.status}')
                oe = np.linalg.norm(odom[:2] - truth[:2])
                ee = np.linalg.norm(ekf[:2] - truth[:2])
                he = np.degrees(wrap_angle(ekf[2] - truth[2]))
                rows.append([t, *truth, *odom, *ekf, *np.diag(P),
                             *(z if z is not None else [np.nan, np.nan, np.nan]),
                             *innovation, nis, int(accepted)])
                if t >= next_print:
                    print(f'{t:4.1f}   {100*oe:12.3f}   {100*ee:11.3f}   {he:18.3f}  {status}')
                    next_print += 1.0
                viewer.sync()
                remaining = model.opt.timestep - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        data.ctrl[motors] = 0
        camera.close()
    if not rows:
        print('No simulation steps completed.')
        return
    print('\nFINAL COMPARISON (world frame)')
    print('             x(m)       y(m)     heading(deg)')
    for label, pose in [('Odometry', odom), ('EKF', ekf), ('Actual', truth)]:
        print(f'{label:9s} {pose[0]:10.4f} {pose[1]:10.4f} {np.degrees(pose[2]):14.3f}')
    print(f'Initial position error: {100*np.linalg.norm(INITIAL_ESTIMATE[:2]-initial_truth[:2]):.3f} cm')
    for label, pose in [('Odometry', odom), ('EKF', ekf)]:
        print(f'{label}: position error={100*np.linalg.norm(pose[:2]-truth[:2]):.3f} cm; '
              f'heading error={np.degrees(wrap_angle(pose[2]-truth[2])):+.3f} deg')
    print('Camera counts:', counts)
    print('EKF standard deviations [m, m, deg]:',
          np.sqrt(P[0,0]), np.sqrt(P[1,1]), np.degrees(np.sqrt(P[2,2])))
    if counts['accepted'] == 0:
        print('NO CAMERA UPDATES: this run does not validate fusion.')
    run_dir = ROOT / 'results' / ('ekf_test_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / 'ekf_log.csv'
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s','true_x','true_y','true_heading_rad',
                         'odom_x','odom_y','odom_heading_rad','ekf_x','ekf_y','ekf_heading_rad',
                         'P_xx','P_yy','P_heading','camera_range_m','camera_bearing_rad','camera_heading_rad',
                         'range_innovation_m','bearing_innovation_rad','heading_innovation_rad','nis','accepted'])
        writer.writerows(rows)
    print('Log saved:', output)
    if camera_heading_errors:
        errors = np.array(camera_heading_errors)
        print(f'CAMERA HEADING DIAGNOSTIC: mean={errors.mean():+.3f} deg; '
              f'RMSE={np.sqrt(np.mean(errors**2)):.3f} deg; '
              f'max absolute={np.max(np.abs(errors)):.3f} deg')
    print('Marker orientation is now used. Remaining error must be measured, not assumed away.')
    print('Q/R are provisional; correlated/bias-prone camera errors still need characterisation.')


if __name__ == '__main__':
    main()
