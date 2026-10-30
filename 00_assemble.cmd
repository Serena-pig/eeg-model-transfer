@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"
where py >nul 2>&1
if errorlevel 1 goto plain_python
py -3 assemble_package.py
if errorlevel 1 goto failed
goto completed
:plain_python
python assemble_package.py
if errorlevel 1 goto failed
:completed
echo COMPLETE. The repository root also contains the ready submission_rank16.xlsx.
pause
exit /b 0
:failed
echo Assembly failed. Keep this error output. No result was submitted.
pause
exit /b 1
