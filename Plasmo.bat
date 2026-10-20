@echo off
rem Abre Plasmo desde el código fuente (la primera vez instala las dependencias).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  where uv >nul 2>nul || (echo Falta uv. Ejecuta primero: powershell -ExecutionPolicy Bypass -File scripts\instalar.ps1 & pause & exit /b 1)
  uv sync --no-dev || (pause & exit /b 1)
)
start "" ".venv\Scripts\pythonw.exe" -m plasmo %*
