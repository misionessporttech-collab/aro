@echo off
setlocal
title Aro - muestra de la panoramica original
set "ARO=%USERPROFILE%\Aro"
set "PATH=%ARO%\herramientas\ffmpeg\bin;%PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest https://raw.githubusercontent.com/misionessporttech-collab/aro/main/procesador/windows/muestra2.py -OutFile $env:TEMP\aro_muestra2.py"
"%ARO%\entorno\Scripts\python.exe" "%TEMP%\aro_muestra2.py"
echo.
pause
