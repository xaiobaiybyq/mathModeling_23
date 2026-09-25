param([string]$DataRoot = (Join-Path $PSScriptRoot '../E题'))
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw '请先按 README 建立 .venv 并安装依赖。' }
& $pythonExe (Join-Path $PSScriptRoot 'test_alignment.py')
if ($LASTEXITCODE -ne 0) { throw '算法检查失败' }
& $pythonExe (Join-Path $PSScriptRoot 'pipeline.py') --data-root $DataRoot --resume
if ($LASTEXITCODE -ne 0) { throw '特征提取存在失败样本，请查看日志' }
& $pythonExe (Join-Path $PSScriptRoot 'summarize.py') --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw '结果验证失败' }
& $pythonExe (Join-Path $PSScriptRoot 'package_results.py')
if ($LASTEXITCODE -ne 0) { throw '打包失败' }
