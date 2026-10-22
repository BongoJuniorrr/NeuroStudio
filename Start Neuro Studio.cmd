@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap-uv.ps1 %*
if errorlevel 1 (
 echo Khong the khoi dong. Xem README.md hoac User Guide trong ung dung.
 pause
)
