@echo off
setlocal
title Aro - video panoramico con nombres (no cierres esta ventana)
set "ARO=%USERPROFILE%\Aro"
set "PATH=%ARO%\herramientas\ffmpeg\bin;%PATH%"
"%ARO%\entorno\Scripts\python.exe" "%ARO%\programa\windows\run_local.py" "%ARO%\partidos" --video
echo.
pause
