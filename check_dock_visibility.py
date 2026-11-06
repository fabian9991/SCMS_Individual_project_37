"""Separate static camera-detection fixture. Does not run or alter the mission.
Use from the project root: python check_dock_visibility.py
This tests detection only, not PnP, EKF gates or closed-loop docking.
"""
from pathlib import Path
import mujoco,cv2,numpy as np
m=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent/'models/robotis_tb3/scene_assignment_plate.xml'));d=mujoco.MjData(m)
m.vis.global_.offwidth=2560;m.vis.global_.offheight=1920
p=cv2.aruco.DetectorParameters();p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX;p.cornerRefinementWinSize=8;p.relativeCornerRefinmentWinSize=.5;p.cornerRefinementMaxIterations=30;p.cornerRefinementMinAccuracy=.001
det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),p)
with mujoco.Renderer(m,1920,2560) as r:
 for gap in [.647,.55,.50,.45,.42,.40]:
  hits=0; count=0
  for y in [-.03,0,.03]:
   for yaw in [-3,0,3]:
    mujoco.mj_resetData(m,d);d.qpos[:2]=[-8.697+gap,y];h=np.pi+np.deg2rad(yaw);d.qpos[3:7]=[np.cos(h/2),0,0,np.sin(h/2)];mujoco.mj_forward(m,d)
    r.update_scene(d,camera='front_camera');im=r.render();cs,ids,_=det.detectMarkers(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY));count+=1;hits+=int(ids is not None and 0 in ids)
  print(gap,hits,count,flush=True)
