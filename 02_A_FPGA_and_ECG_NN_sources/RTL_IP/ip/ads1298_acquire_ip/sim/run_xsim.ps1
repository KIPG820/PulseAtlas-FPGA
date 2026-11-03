$ErrorActionPreference = 'Stop'
$ipRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vivadoBin = 'D:\Xilinx\Vivado\2024.2\bin'
$simDir = Join-Path $env:TEMP 'ecg_ads1298_xsim'
New-Item -ItemType Directory -Path $simDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ipRoot 'rtl\ads1298_spi_master.sv'),
                            (Join-Path $ipRoot 'rtl\ads1298_acquire_ip.sv'),
                            (Join-Path $PSScriptRoot 'tb_ads1298_acquire.sv') -Destination $simDir -Force

# XSim's bundled Tcl fails to parse the Chinese Desktop path on this host.
# Map an unused drive to the ASCII temporary directory for the simulator run.
$drive = @('X:', 'Y:', 'Z:', 'Q:') | Where-Object { -not (Test-Path "$_\") } | Select-Object -First 1
if (-not $drive) { throw 'No free temporary drive letter for XSim.' }
try {
    & subst $drive $simDir
    if ($LASTEXITCODE -ne 0) { throw 'subst failed.' }
    Set-Location "$drive\"
    & (Join-Path $vivadoBin 'xvlog.bat') -sv ads1298_spi_master.sv ads1298_acquire_ip.sv tb_ads1298_acquire.sv
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed.' }
    & (Join-Path $vivadoBin 'xelab.bat') tb_ads1298_acquire -s tb_ads1298_acquire_sim
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed.' }
    $result = & (Join-Path $vivadoBin 'xsim.bat') tb_ads1298_acquire_sim -runall 2>&1
    $result | ForEach-Object { Write-Output $_ }
    if ($LASTEXITCODE -ne 0 -or ($result -join "`n") -notmatch 'PASS: idle hot-plug detection, stream start/stop') {
        throw 'XSim failed or did not reach the PASS check.'
    }
} finally {
    Set-Location $PSScriptRoot
    & subst $drive /D | Out-Null
}
