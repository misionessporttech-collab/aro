@echo off
setlocal
title Aro - cortar muestra de prueba
set "ARO=%USERPROFILE%\Aro"
set "PATH=%ARO%\herramientas\ffmpeg\bin;%PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest https://raw.githubusercontent.com/misionessporttech-collab/aro/main/procesador/windows/muestra.py -OutFile $env:TEMP\aro_muestra.py"
"%ARO%\entorno\Scripts\python.exe" "%TEMP%\aro_muestra.py"
echo.
pause
