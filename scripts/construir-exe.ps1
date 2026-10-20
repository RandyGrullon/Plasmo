# Compila dist\Plasmo\Plasmo.exe (carpeta portable) y crea el acceso directo.
#   powershell -ExecutionPolicy Bypass -File scripts\construir-exe.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

uv sync --group dev
uv run python scripts/hacer_icono.py
uv run pyinstaller --noconfirm --clean plasmo.spec

$exe = Join-Path $root "dist\Plasmo\Plasmo.exe"
if (-not (Test-Path $exe)) { throw "No se generó $exe" }

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "Plasmo.lnk"))
$lnk.TargetPath = $exe
$lnk.WorkingDirectory = Split-Path $exe
$lnk.Save()

Compress-Archive -Path (Join-Path $root "dist\Plasmo") -DestinationPath (Join-Path $root "dist\Plasmo-windows.zip") -Force
Write-Host "Listo: $exe (y dist\Plasmo-windows.zip para copiarlo a otra PC)" -ForegroundColor Green
