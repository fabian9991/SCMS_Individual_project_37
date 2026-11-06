LQR RECOVERY TESTS — FIXED BASELINE

Purpose
-------
Measure how the existing controller recovers before changing its gains.
This is a separate test runner. It does not edit navigate_all_goals.py,
test_ekf.py, the planner, or the model XML files.

Setup
-----
Copy test_lqr_recovery.py into your project root, beside:
  navigate_all_goals.py (capital N is also supported)
  test_ekf.py
  lqr_diagnostics.py
  test_wheel_response.py
The last two files are from the previous lqr_updates package, which you used
for the latest wheel-response and mission tests. Keep that instrumentation.
The models/ and calibration/ folders must remain in their existing locations.

Keep the baseline configuration:
  ENCODER_NOISE_ENABLED = True
  ENCODER_NOISE_M_PER_SQRT_M = 0.005
  USE_WHEEL_TRAVEL_Q = True
  SLIP_PLATE_ENABLED = False
  R_CAMERA standard deviations = 0.02 m, 0.5 degrees, 5 degrees
Keep the LQR weights, wheel servo, control period and speed limits unchanged.
The --seed argument selects the noise sequence for this test runner; you do
not need to edit ENCODER_NOISE_SEED in the navigation file for these tests.

Run from the project root in Windows PowerShell
----------------------------------------------
First run all seven cases with the viewer:

  .\.venv\Scripts\python.exe .\test_lqr_recovery.py --seed 37 --viewer

Leave the viewer open. The robot resets between cases automatically.
Then repeat with two other noise sequences; these commands omit the viewer:

  .\.venv\Scripts\python.exe .\test_lqr_recovery.py --seed 39
  .\.venv\Scripts\python.exe .\test_lqr_recovery.py --seed 42

The camera renderer still runs without the viewer. These commands use the
same MuJoCo/OpenCV environment as the successful mission. Each case has a
60-second simulation timeout; headless runs do not deliberately wait for
real time. Rendering can still make a run slow.

To inspect just one case, for example:

  .\.venv\Scripts\python.exe .\test_lqr_recovery.py --seed 37 --case corner --viewer

Cases and meaning
-----------------
on_path              : Starts on the straight, facing along it.
sideways_left        : Starts 5 cm left of the straight, parallel to it.
sideways_right       : Starts 5 cm right of the straight, parallel to it.
heading_left         : Starts on the straight with a +10-degree heading error.
heading_right        : Starts on the straight with a -10-degree heading error.
alignment_transition : Starts on the straight with a +30-degree heading error.
corner               : Tracks a 1 m straight, then a perpendicular 0.4 m segment.

The first five cases start in TRACK. In particular, the +/-10-degree cases
must exercise LQR correction while moving; starting them in ALIGN would
mostly test the separate rotation controller instead.

alignment_transition starts in ALIGN and uses the existing 4-degree
threshold to enter TRACK. The corner uses the normal segment handover and
ALIGN behaviour. Watch for a sudden wheel-command change at those handovers.

The straight is 1.5 m long, from world (-4, 0) towards (-5.5, 0).
The corner follows (-4, 0) -> (-5, 0) -> (-5, +0.4). These fixtures use the
open corridor in the supplied scene. The reference line stays fixed, so a
planner cannot move the route underneath the robot and hide tracking error.
LiDAR stopping remains active. A LiDAR stop ends that case and is recorded.
The seed 41 full-mission planning stall is not fixed or retested by this
controller-only fixture; the planner is not used here.

Known offset does not mean deliberately wrong estimation
-------------------------------------------------------
Each case declares a starting pose. That same declared pose sets up the
physical test fixture and initialises the EKF. The robot begins physically
beside, or rotated relative to, the reference line. We do not put an error
into the estimate while leaving the robot elsewhere.

Simulator truth is never copied into the estimator or used for control.
It is read only for scoring, including any small change during settling.
There are no physical pose resets during a trial, only between trials.
The camera and its mission gates are retained. These short routes are
expected to remain outside the 3 m range gate; accepted updates are logged.

Outputs
-------
Each invocation creates a unique folder:
  results/lqr_recovery_seed_<seed>_<timestamp>/

At its top level:
  run_config.json        Settings, gains and source-file hashes.
  all_cases_summary.csv  One row per segment (eight rows for seven cases).

Each case has:
  case_config.json       Declared start, reference route and initial scoring.
  controller.csv         Estimated/actual tracking errors and wheel commands.
  controller_events.csv Mode changes, segment handover, camera updates, ending.
  tracking_summary.csv  Tracking RMS/max errors and diagnostic recovery time.
  summary.txt           Outcome, endpoint distances and camera counts.
  error.txt             Present only if an exception occurs during the trial.

Logs are flushed during the trial. Ctrl+C or an exception during the trial
also attempts to write a partial summary. Abruptly killing Python can still
prevent final summaries; keep any available logs.

How to assess the results
------------------------
1. Check outcome and initial actual offset first. Retain unfinished cases.
2. Plot estimated and actual cross-track/heading error against time. The
   actual error is the physical result; the estimated error is what LQR saw.
3. Compare each offset case with on_path for the same seed. Examine recovery,
   overshoot, residual error and endpoint error, not just one final number.
4. Inspect actuator limiting and wheel rates around ALIGN -> TRACK and the
   corner. Rates describe the previous interval; commands apply to the next
   interval. controller.csv explicitly records those timing relationships.
5. Compare seeds before deciding whether a change helps consistently.

Recovery entry time is the first time cross-track error is within +/-1 cm
AND heading error within +/-2 degrees for one uninterrupted second of
active tracking. It is measured from case start, separately for each segment
and for estimated/actual pose. Blank means that interval was not observed;
it does not mean zero seconds. Later departures remain possible and must
be checked in the plots. The short second corner segment may provide less
time to settle. This is a diagnostic band, not a mission pass/fail threshold.

Endpoint arrival uses the existing Tracker's 2 cm estimated-distance
tolerance. Actual distance is reported without a 5 cm pass/fail cutoff.
For corner summary rows, endpoint arrival fields refer to the final endpoint
of the whole case; segment-specific errors and recovery times remain separate.

Next decision
-------------
Do not change the LQR weights solely because actual position error exceeds
estimated tracking error. That difference may come from localisation.
If the transition cases show a problematic speed jump, trial a forward-speed
ramp after alignment as a separate change, then repeat exactly these tests.
Upload the complete three result folders for analysis and the next write-up.

Verification of this supplied script
-----------------------------------
Syntax, geometry/signs, recovery-band logic, seven-case logging, corner
handover and partial-result saving were checked using mocked interfaces.
MuJoCo/OpenCV physics runs were not available in the authoring environment.
Those checks are software checks, not measured recovery performance. The
new tests remain marked as proposed in the Word document until you run them.
