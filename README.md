# SCMS_Individual_project_37

Open the virtual environment:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Install the dependencies:
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

Open World:
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=".\models\robotis_tb3\scene_assignment.xml"



Opens and runs the world with no prior steps:
.\.venv\Scripts\python.exe .\navigate_all_goals.py

Once the program finishes it's run it closes automatically.