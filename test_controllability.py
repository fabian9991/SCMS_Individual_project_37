"""Run beside navigate_all_goals.py. Use --rank-only to skip MuJoCo."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from navigate_all_goals import (
    DT_CONTROL, START_ESTIMATE, START_P, Tracker, WheelServo,
    evaluation_pose, predict_ekf, wheel_targets, wrap_angle,
)

ROOT = Path(__file__).resolve().parent


def rank_check(speeds):
    # Same exact zero-order-hold tracking model as navigate_all_goals.lqr_gain.
    dt = DT_CONTROL
    print("Speed (m/s)   Rank of [B, AB]   Expected")
    for v in [0.0, *speeds]:
        A = np.array([[1.0, v * dt], [0.0, 1.0]])
        B = np.array([[0.5 * v * dt**2], [dt]])
        rank = np.linalg.matrix_rank(np.column_stack((B, A @ B)))
        expected = 1 if v == 0 else 2
        print(f"{v:10.3f}   {rank:15}   {expected}")
        if rank != expected:
            raise RuntimeError("Tracking model failed the expected rank check.")


def drive(tracker):
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_path(str(ROOT / "models/robotis_tb3/scene_assignment.xml"))
    data = mujoco.MjData(model)
    names = ("wheel_left", "wheel_right")
    addresses = [model.joint(n).qposadr[0] for n in names]
    motors = [model.actuator(n).id for n in names]
    base = model.body("base").id
    dt = model.opt.timestep
    for _ in range(round(1.0 / dt)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    pose, P = START_ESTIMATE.copy(), START_P.copy()
    heading = pose[2]
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-forward[1], forward[0]])
    # Leave the robot at its known XML start. Shift the reference line
    # 5 cm to its right so the starting cross-track error is +5 cm.
    origin = pose[:2] - 0.05 * left
    tracker.route = np.array([origin, origin + 1.5 * forward])
    servo = WheelServo()
    enc = data.qpos[addresses].copy()
    control_enc = enc.copy()
    rates = np.zeros(2)
    start = last_control = data.time
    rows = []
    while True:
        t = data.time - start
        elapsed = data.time - last_control
        if not rows or elapsed >= DT_CONTROL - 1e-10:
            if rows:
                rates = (enc - control_enc) / elapsed
            v, omega, _, _ = tracker.command(pose)
            stop = tracker.mode in ("DONE", "OVERSHOOT") or t >= 40
            data.ctrl[motors] = servo.command(wheel_targets(v, omega), rates,
                                               elapsed or DT_CONTROL, stop)
            # Actual pose enters only the score, after computing the command.
            mujoco.mj_forward(model, data)
            actual = evaluation_pose(data, base)
            rows.append((t, left @ (pose[:2] - origin), left @ (actual[:2] - origin),
                         np.degrees(wrap_angle(actual[2] - heading)), v, omega))
            if stop:
                break
            last_control = data.time
            control_enc = enc.copy()
        mujoco.mj_step(model, data)
        new_enc = data.qpos[addresses].copy()
        pose, P = predict_ekf(pose, P, new_enc - enc, dt)
        enc = new_enc

    folder = ROOT / "results" / ("controllability_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    with (folder / "tracking.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "estimated_side_m", "actual_side_m",
                         "actual_heading_error_deg", "command_m_s", "command_rad_s"])
        writer.writerows(rows)
    times, estimated, actual = np.array(rows).T[:3]
    fig, ax = plt.subplots()
    ax.plot(times, actual * 100, label="Actual")
    ax.plot(times, estimated * 100, "--", label="Encoder estimate")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="Time (s)", ylabel="Signed sideways error (cm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "tracking.png")
    plt.close(fig)
    status = tracker.mode if tracker.mode in ("DONE", "OVERSHOOT") else "TIMEOUT"
    print(f"Tracking result: {status}")
    print(f"Actual sideways error: start {100*actual[0]:+.2f} cm, end {100*actual[-1]:+.2f} cm")
    print(f"Actual sideways RMSE: {100*np.sqrt(np.mean(actual**2)):.2f} cm")
    print(f"Saved CSV and plot in {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-only", action="store_true")
    args = parser.parse_args()
    tracker = Tracker()
    rank_check(tracker.speeds)
    if not args.rank_only:
        drive(tracker)
