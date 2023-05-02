@echo off

REM Set the path to your virtual environment's activate script
set VENV_PATH=venv\Scripts\activate.bat

REM Open three command prompt instances and activate the virtual environment on each one
start "Client 1" cmd /k "%VENV_PATH% && python client.py"
start "Client 2" cmd /k "%VENV_PATH% && python client.py"

exit