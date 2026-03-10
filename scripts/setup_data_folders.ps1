# PowerShell script to create data directory structure
# Run this from the capstone project root directory

Write-Host "Creating data directory structure..." -ForegroundColor Green

# Create main data directory if it doesn't exist
New-Item -Path "data" -ItemType Directory -Force | Out-Null

# Create subdirectories
New-Item -Path "data/raw" -ItemType Directory -Force | Out-Null
New-Item -Path "data/interim" -ItemType Directory -Force | Out-Null
New-Item -Path "data/processed" -ItemType Directory -Force | Out-Null
New-Item -Path "data/features" -ItemType Directory -Force | Out-Null

Write-Host "✓ data/raw - Created (upload your Scopus and SciVal files here)" -ForegroundColor Cyan
Write-Host "✓ data/interim - Created" -ForegroundColor Cyan
Write-Host "✓ data/processed - Created" -ForegroundColor Cyan
Write-Host "✓ data/features - Created" -ForegroundColor Cyan

Write-Host "`nData directory structure created successfully!" -ForegroundColor Green
Write-Host "Upload your data files to: data/raw/" -ForegroundColor Yellow
