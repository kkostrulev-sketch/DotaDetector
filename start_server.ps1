# Запуск API-сервиса DOTADetector
# После старта откройте в браузере: http://localhost:8000

Set-Location $PSScriptRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "Сервер запускается..." -ForegroundColor Cyan
Write-Host "Откройте в браузере:  http://localhost:8000" -ForegroundColor Green
Write-Host "Документация API:     http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Проверка здоровья:    http://localhost:8000/health" -ForegroundColor Green
Write-Host ""
Write-Host "НЕ используйте 0.0.0.0 в браузере — это адрес для сервера, не для пользователя." -ForegroundColor Yellow
Write-Host "Остановка: Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

uvicorn service.app.main:app --host 0.0.0.0 --port 8000