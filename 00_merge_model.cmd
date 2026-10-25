@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 merge_model.py
if errorlevel 1 (
 echo 合并失败，请保留上面的报错内容。
 pause
 exit /b 1
)
pause
