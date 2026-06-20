# Запуск Gradio GUI для DOTADetector
# Откройте в браузере: http://localhost:7860

Set-Location $PSScriptRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "Gradio GUI запускается..." -ForegroundColor Cyan
Write-Host "Откройте в браузере:  http://localhost:7860" -ForegroundColor Green
Write-Host ""
Write-Host "НЕ используйте 0.0.0.0 в браузере — это адрес для сервера." -ForegroundColor Yellow
Write-Host "Остановка: Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

python service/gradio_app.py