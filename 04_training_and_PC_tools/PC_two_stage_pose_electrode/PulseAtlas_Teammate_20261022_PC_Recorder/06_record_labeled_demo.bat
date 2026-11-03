@echo off
setlocal
cd /d "%~dp0"
set PULSEATLAS_DISABLE_IDENTITY_GUARD=1
set PULSEATLAS_DISABLE_TEMPORAL=
set PULSEATLAS_PREVIOUS_BASELINE=
set "PA_PY=python"
if defined CONDA_PREFIX set "PA_PY=%CONDA_PREFIX%\python.exe"
if not defined CONDA_PREFIX if exist "%USERPROFILE%\.conda\envs\yolo\python.exe" set "PA_PY=%USERPROFILE%\.conda\envs\yolo\python.exe"
"%PA_PY%" verify_package.py
if errorlevel 1 exit /b 1
"%PA_PY%" app/camera_trial.py --demo-ui
endlocal
pause
