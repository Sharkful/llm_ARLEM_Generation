# setup_windows.ps1
Write-Host "Creating Virtual Environment..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Activating Environment and Installing Requirements..." -ForegroundColor Cyan
# Use the direct path to pip to avoid activation issues
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Setup Complete! To activate your environment, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green