$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$run = Join-Path $PSScriptRoot 'sim_run'
New-Item -ItemType Directory -Force -Path $run | Out-Null
python (Join-Path $PSScriptRoot 'build_variant.py')
if ($LASTEXITCODE -ne 0) { throw 'Variant generation failed' }
python (Join-Path $PSScriptRoot 'build_testbench.py')
if ($LASTEXITCODE -ne 0) { throw 'Testbench generation failed' }
$ip = Join-Path $root 'bld\ecg_nn_r9\ip_repo_ecg\ecg_qrs_ip\src'
$bin = 'D:\Xilinx\Vivado\2024.2\bin'
Push-Location $run
try {
    & "$bin\xvlog.bat" --sv -i $ip "$ip\ecg_frame_pkg.sv" "$ip\ecg_filter_stages.sv" "$PSScriptRoot\ecg_qrs_detector_reference.sv" "$PSScriptRoot\ecg_qrs_detector_ram.sv" "$PSScriptRoot\tb_qrs_storage_equiv.sv" *> compile.log
    if ($LASTEXITCODE -ne 0) { Get-Content compile.log -Tail 30; throw 'Compile failed' }
    & "$bin\xelab.bat" tb_qrs_storage_equiv -s qrs_storage_equiv -debug typical --timescale 1ns/1ps *> elaborate.log
    if ($LASTEXITCODE -ne 0) { Get-Content elaborate.log -Tail 30; throw 'Elaboration failed' }
    & "$bin\xsim.bat" qrs_storage_equiv -runall *> simulate.log
    Get-Content simulate.log -Tail 25
    if ($LASTEXITCODE -ne 0 -or !(Select-String -Path simulate.log -Pattern 'PASS all QRS storage equivalence cases' -Quiet)) { throw 'Simulation failed' }
} finally {
    Pop-Location
}
