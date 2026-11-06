# SCMS Individual Project 37

## Track A — mobile robot visual waypoint navigation

This project uses a TurtleBot3 Waffle Pi in MuJoCo to visit three storage bays and return to an ArUco-marked charging station. Wheel odometry and camera measurements are fused by an EKF. LiDAR supports obstacle mapping and route planning, and an LQR follows the planned segments using the estimated pose. Final docking uses low-speed visual pose feedback.

Simulator position and heading are used for evaluation, not navigation feedback. Docking is to a stand-off position, not electrical charging contact.

- **Repository:** [SCMS_Individual_project_37](https://github.com/fabian9991/SCMS_Individual_project_37)
- **Main test:** `navigate_all_goals.py`
- **Demo video:** [Watch the project demonstration on YouTube](https://youtu.be/mWIeYK-iH-M?si=dDp2DcB_gHl5Da40)

## 1. Open the correct folder

Run the main test from the **repository root**, named `SCMS_Individual_project_37`. This is the folder containing `navigate_all_goals.py`, `requirements.txt`, `models` and `calibration`.

Open PowerShell and enter the following, replacing the example path with the location of your own checkout:

```powershell
cd "C:\path\to\SCMS_Individual_project_37"
```

Check the location and main script:

```powershell
Get-Location
Test-Path .\navigate_all_goals.py
```

The second command should print `True`. Keep the terminal open while the simulation is running.

## 2. First-time installation

The project was developed using Windows and Python 3.13. A graphical desktop and working OpenGL support are required for the MuJoCo viewer and camera rendering. Install Python 3.13 and Git before the following steps.

Create the virtual environment and install the pinned dependencies from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

If `.venv` already exists and works, skip its creation. The commands below call its Python interpreter directly, so activating the environment or changing PowerShell's execution policy is unnecessary.

The supplied dependency file pins MuJoCo, NumPy, OpenCV contrib and supporting packages. OpenCV contrib supplies the ArUco functionality.

## 3. Run the main test

From the repository root, enter:

```powershell
.\.venv\Scripts\python.exe .\navigate_all_goals.py
```

The MuJoCo window opens and the terminal displays the demo menu. The mission does **not** start automatically. The robot waits at the launch position while you choose a condition.

### Select a condition

| Enter | Encoder noise | Slip plate behaviour |
|---|---|---|
| `1` | No deliberately added encoder noise | Normal traction |
| `2` | Added encoder noise | Normal traction |
| `3` | Added encoder noise | Low-friction slip plate enabled |

The physical floor insert remains present in all three cases; the setting changes its friction. Mode 1 is not a perfect, error-free simulation: camera measurements and the physical model can still differ from the ideal motion model.

For example, to run the noise-and-slip case:

1. Click the **terminal**, type `3`, then press **Enter**.
2. Type `s`, then press **Enter**.
3. Watch the robot visit the bays and return to the charger.

The default initial preset is mode 2, with seed 37 and a 0.50 m base-to-marker docking target. The terminal prints the selected settings before starting.

### Terminal controls

**Every command requires Enter and terminal focus.** These are terminal commands, not shortcuts in the MuJoCo window. Type the letter itself; for example, use `p` + Enter, not Ctrl+P.

| Command | Action |
|---|---|
| `1`, `2`, `3` | Select a condition while the menu is ready, before starting. |
| `s` | Start the selected trial, or resume a paused trial. |
| `p` | Pause the active trial. Simulation time and physics are frozen. |
| `r` | Save the active trial if necessary, reset to the launch origin, clear trails and return to the settings menu. |
| `q` | Save the active trial if necessary and close the simulator. |
| `h` | Display the available controls. |

Settings cannot be changed during an active trial. Use `r` to return to the menu before choosing a different condition. Reset creates a fresh estimator, map, controller and trial; it does not command the robot to drive back to the origin.

### Run all three demo cases without editing files

Run mode 1 first by entering `1`, then `s`, pressing Enter after each command. When it finishes, enter `r`, then `2`, then `s` to run mode 2. Repeat with `r`, `3`, then `s` for mode 3. Finally, enter `q` to close.

The viewer stays open after a completed interactive run, holding the final pose and trails. This is intentional. Look for the final outcome and `RUN FINISHED` message in the terminal, then choose reset or close. A result that reports interruption or failure must not be counted as a successful mission.


## 4. Results and troubleshooting

Each trial saves a timestamped directory under `results/`. The terminal prints its full path. Depending on the trial, outputs include the trajectory, goal errors, estimator comparison, final pose report, status, selected demo settings, LiDAR map and collision/slip summaries. Resetting preserves the previous trial's saved results.

| Issue | Check |
|---|---|
| Python cannot find `navigate_all_goals.py` | Return to the repository root and check `Test-Path .\navigate_all_goals.py`. |
| `.venv\Scripts\python.exe` is missing | Create the virtual environment using the installation commands. |
| Missing Python module | Install `requirements.txt` using the same `.venv` interpreter used to run the test. |
| Camera calibration or model assets are missing | Restore the complete supplied project directory structure. |
| A terminal command appears to do nothing | Focus the terminal and press Enter after the command. |
| The robot is waiting at startup | Select `1`, `2` or `3`, then enter `s`. |
| The window remains open after completion | Enter `r` for another trial or `q` to close. |
| `WAIT CAMERA` is displayed | Docking is waiting for a fresh accepted visual observation. This is not a completed dock. |

Opening the XML in the standalone MuJoCo viewer only displays the scene. It does not run the EKF, planner or controller. Use `navigate_all_goals.py` for the full test.

## 5. Individual contribution

**Fabian — individual project.** Covered the MuJoCo environment, camera calibration and accuracy testing, and integration of wheel odometry, EKF localisation, LQR tracking and LiDAR navigation. I also set up encoder-noise and slip-plate tests, analysed navigation and docking errors, documented the motion model and initial controllability tests, and prepared the project report and website.

**John — individual project.** Responsible for the testing and refinement. Motion-model validation, controllability and observability analysis, camera calibration and measurement evaluation, EKF uncertainty tuning and repeated-seed comparisons, LQR and wheel-response evaluation, and investigation of tracking and planning failures. Contributions also included collision and slip evaluation, docking refinement and camera-detection troubleshooting, path visualisation, terminal start/pause/reset controls, the three-condition demonstration, quantitative results analysis, and preparation of the report and README.

## 6. Generative AI declaration

Google Gemini was used to explain modelling and control concepts, discuss implementation choices, debugging, interpret supplied results, and revise project documentation. VS Codes inbuild copilot was used for debugging and terminal controls. As well as automatic code suggests were used when provided once verified they were correct. 

Reported numerical results come from saved simulation outputs. I remain responsible for checking the submitted code, calculations and claims, and for explaining the integrated system in the viva. 

## 7. References and source acknowledgements

### Simulation and robot assets

- MuJoCo. (n.d.). *MuJoCo project*. https://mujoco.org/
- MuJoCo. (n.d.). *Documentation*. https://mujoco.readthedocs.io/
- MuJoCo. (n.d.). *Python bindings and viewer*. https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo. (n.d.). *XML reference*. https://mujoco.readthedocs.io/en/latest/XMLreference.html
- MuJoCo. (n.d.). *Modeling guide*. https://mujoco.readthedocs.io/en/latest/modeling.html
- MuJoCo. (n.d.). *Modeling guide, version 3.17.0*. https://mujoco.readthedocs.io/en/3.17.0/modeling.html
- MuJoCo. (n.d.). *Camera conventions*. https://mujoco.readthedocs.io/en/stable/modeling.html#cameras
- MuJoCo. (n.d.). *API reference, version 2.1.5*. https://mujoco.readthedocs.io/en/2.1.5/APIreference.html
- Google DeepMind. (n.d.). *MuJoCo public API header*. https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h
- ROBOTIS. (n.d.). *TurtleBot3 overview*. https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/
- ROBOTIS. (n.d.). *ROBOTIS MuJoCo Menagerie*. https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- ROBOTIS. (n.d.). *TurtleBot3 model assets*. https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main/robotis_tb3
- ROBOTIS. (n.d.). *TurtleBot3 Waffle Pi XML model*. https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/blob/main/robotis_tb3/turtlebot3_waffle_pi.xml

### Camera calibration and marker detection

- OpenCV. (n.d.). *OpenCV project*. https://opencv.org/
- OpenCV. (n.d.). *Documentation*. https://docs.opencv.org/
- OpenCV. (n.d.). *Camera calibration and 3D reconstruction, version 4.5.2*. https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html
- OpenCV. (n.d.). *ArUco module, version 4.7.0*. https://docs.opencv.org/4.7.0/de/d67/group__objdetect__aruco.html
- OpenCV. (n.d.). *ArUco module, version 4.11.0*. https://docs.opencv.org/4.11.0/de/d67/group__objdetect__aruco.html
- OpenCV. (n.d.). *ArucoDetector class, version 4.9.0*. https://docs.opencv.org/4.9.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html
- OpenCV. (n.d.). *DetectorParameters, version 5.0*. https://docs.opencv.org/5.0/main_modules/structcv_1_1aruco_1_1DetectorParameters.html
- OpenCV. (n.d.). *Perspective-n-Point pose computation, version 4.11.0*. https://docs.opencv.org/4.11.0/d5/d1f/calib3d_solvePnP.html
- OpenCV. (n.d.). *Perspective-n-Point pose computation, version 4.12.0*. https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html
- OpenCV. (n.d.). *Perspective-n-Point documentation link cited during development*. https://docs.opencv.org/doc/doxygen/html/d5/d1f/calib3d_solvePnP.html

### Numerical tools and project setup

- NumPy. (n.d.). *NumPy documentation*. https://numpy.org/doc/
- Git. (n.d.). *Git for Windows installation*. https://git-scm.com/install/windows
- Fabian. (n.d.). *SCMS project source, test branch*. https://github.com/fabian9991/SCMS_Individual_project_37/tree/test
- SCMS project. (n.d.). *Bundled robot-model licence*. https://github.com/fabian9991/SCMS_Individual_project_37/blob/test/models/robotis_tb3/LICENSE

### Teaching material

- R. Falque. (n.d.). *MuJoCo UR3 tutorial repository*. https://github.com/rFalque/mujoco_UR3_tutorial
- Nerfies contributors. (n.d.). *Project website template repository*. https://github.com/nerfies/nerfies.github.io

Course materials also include the supplied Sensors and Control tutorial notes and solutions. MuJoCo provides simulation and sensor rendering, OpenCV provides calibration and marker pose estimation, and NumPy provides numerical operations. These libraries are not original project contributions.
