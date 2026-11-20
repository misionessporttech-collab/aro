@echo off
setlocal
title Aro - instalacion del procesador
echo.
echo  ===========================================================
echo   Aro by Misiones Sport Tech - instalacion del procesador
echo  ===========================================================
echo.
echo  Se instala todo en %USERPROFILE%\Aro (no toca nada de Windows).
echo  Tarda entre 10 y 20 minutos segun la conexion. Descarga unos 3 GB.
echo.

set "ARO=%USERPROFILE%\Aro"
set "TOOLS=%ARO%\herramientas"
mkdir "%ARO%" 2>nul
mkdir "%TOOLS%" 2>nul
mkdir "%ARO%\partidos" 2>nul

echo [1/6] Revisando la placa de video...
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if errorlevel 1 (
  echo.
  echo  No encuentro el driver de NVIDIA. Instala o actualiza el driver desde
  echo  https://www.nvidia.com/es-la/geforce/drivers/  y volve a ejecutar este archivo.
  pause
  exit /b 1
)

echo [2/6] Descargando el programa de Aro...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP 'aro.zip'; Invoke-WebRequest 'https://github.com/misionessporttech-collab/aro/archive/refs/heads/main.zip' -OutFile $z; $d=Join-Path $env:TEMP 'aro-src'; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive $z $d; $p=Join-Path $env:USERPROFILE 'Aro\programa'; if(Test-Path $p){Remove-Item $p -Recurse -Force}; Copy-Item (Join-Path $d 'aro-main\procesador') $p -Recurse"
if errorlevel 1 goto fallo

echo [3/6] Descargando FFmpeg (lectura de video)...
if not exist "%TOOLS%\ffmpeg\bin\ffmpeg.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP 'ffmpeg.zip'; Invoke-WebRequest 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile $z; $d=Join-Path $env:TEMP 'ffmpeg-x'; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive $z $d; $s=Get-ChildItem $d -Directory | Select-Object -First 1; Move-Item $s.FullName (Join-Path $env:USERPROFILE 'Aro\herramientas\ffmpeg')"
  if errorlevel 1 goto fallo
)

echo [4/6] Descargando uv (instalador de Python)...
if not exist "%TOOLS%\uv\uv.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR=(Join-Path $env:USERPROFILE 'Aro\herramientas\uv'); $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
  if errorlevel 1 goto fallo
)
set "UV=%TOOLS%\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"

echo [5/6] Instalando Python y las librerias (la parte mas larga)...
if not exist "%ARO%\entorno\Scripts\python.exe" "%UV%" venv "%ARO%\entorno" --python 3.12
if errorlevel 1 goto fallo
"%UV%" pip install --python "%ARO%\entorno\Scripts\python.exe" torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto fallo
"%UV%" pip install --python "%ARO%\entorno\Scripts\python.exe" ultralytics opencv-python scipy scikit-learn numpy
if errorlevel 1 goto fallo

echo [6/6] Probando la placa de video desde Python...
"%ARO%\entorno\Scripts\python.exe" -c "import torch;ok=torch.cuda.is_available();print('  Placa de video lista:', torch.cuda.get_device_name(0)) if ok else print('  ATENCION: Python no ve la placa. Actualiza el driver de NVIDIA y volve a ejecutar instalar.bat');raise SystemExit(0 if ok else 1)"
if errorlevel 1 (pause & exit /b 1)

copy /y "%ARO%\programa\windows\procesar.bat" "%ARO%\procesar.bat" >nul

echo.
echo  ===========================================================
echo   Listo. Para procesar un partido:
echo   1. Crea una carpeta dentro de %ARO%\partidos
echo      (por ejemplo: mitre-suardi) y pone adentro la grabacion
echo      panoramica y el respaldo del partido exportado de Aro (.json).
echo   2. Doble clic en %ARO%\procesar.bat
echo  ===========================================================
echo.
explorer "%ARO%"
pause
exit /b 0

:fallo
echo.
echo  Algo fallo en este paso. Revisa la conexion a internet y volve a ejecutar
echo  instalar.bat: retoma lo que falta. Si se repite, mandale a Claude una foto
echo  de esta ventana.
pause
exit /b 1
