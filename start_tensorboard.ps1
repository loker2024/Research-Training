# start_tensorboard.ps1

$LogDir = ".\runs"

# Initialize Conda for PowerShell
conda shell.powershell hook | Out-String | Invoke-Expression

# Activate environment
conda activate myenv

if ($env:CONDA_DEFAULT_ENV -ne "myenv") {
    Write-Error "Failed to activate Conda environment: myenv"
    exit 1
}

Write-Host "Active Conda environment: $env:CONDA_DEFAULT_ENV"
Write-Host "TensorBoard log directory: $LogDir"
Write-Host "TensorBoard URL: http://localhost:6006"

tensorboard --logdir $LogDir --port 6006