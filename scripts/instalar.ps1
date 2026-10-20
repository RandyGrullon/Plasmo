# Instala Plasmo desde el código: uv, dependencias, modelos ONNX livianos y acceso directo.
#   powershell -ExecutionPolicy Bypass -File scripts\instalar.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando uv..." -ForegroundColor Cyan
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    } else {
        powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Instalando dependencias (Python 3.12 incluido)..." -ForegroundColor Cyan
uv sync --no-dev

Write-Host "Descargando modelos de recorte y profundidad (~280 MB)..." -ForegroundColor Cyan
uv run --no-dev python -c "from plasmo.imaging import downloads as d; [print(d.download(k)) for k in ('isnet-general-use', 'depth-anything-v2-small')]"

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "Plasmo.lnk"))
$lnk.TargetPath = Join-Path $root "Plasmo.bat"
$lnk.WorkingDirectory = $root
$lnk.IconLocation = Join-Path $root "assets\plasmo.ico"
$lnk.WindowStyle = 7
$lnk.Save()

Write-Host ""
Write-Host "Listo. Abre 'Plasmo' desde el escritorio." -ForegroundColor Green
Write-Host "Falta la IA (Ollama + ComfyUI)? Mira docs\INSTALAR-IA.md"
