@echo off
setlocal
title Aro - procesando partidos (no cierres esta ventana)
set "ARO=%USERPROFILE%\Aro"
set "PATH=%ARO%\herramientas\ffmpeg\bin;%PATH%"

echo Buscando la version mas nueva del programa...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $z=Join-Path $env:TEMP 'aro.zip'; Invoke-WebRequest 'https://github.com/misionessporttech-collab/aro/archive/refs/heads/main.zip' -OutFile $z; $d=Join-Path $env:TEMP 'aro-src'; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive $z $d; $p=Join-Path $env:USERPROFILE 'Aro\programa'; Remove-Item $p -Recurse -Force; Copy-Item (Join-Path $d 'aro-main\procesador') $p -Recurse; Copy-Item (Join-Path $p 'windows\generar_video.bat') (Join-Path $env:USERPROFILE 'Aro\generar_video.bat') -Force; Write-Host '  Programa actualizado.' } catch { Write-Host '  Sin internet: uso la version que ya estaba.' }"

"%ARO%\entorno\Scripts\python.exe" "%ARO%\programa\windows\run_local.py" "%ARO%\partidos"
echo.
pause
