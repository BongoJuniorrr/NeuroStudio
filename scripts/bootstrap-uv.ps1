$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root '.tools'
$uv = Join-Path $tools 'uv.exe'
if (!(Test-Path $uv)) {
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $manifest = Get-Content (Join-Path $root 'packaging/uv.json') -Raw | ConvertFrom-Json
    $asset = $manifest.files.'windows-x64'
    $archive = Join-Path $tools 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri $asset.url -OutFile $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLower() -ne $asset.digests.sha256) { throw 'UV_CHECKSUM_MISMATCH' }
    $expanded = Join-Path $tools 'unpack'
    Expand-Archive -Force -Path $archive -DestinationPath $expanded
    $binary = Get-ChildItem $expanded -Filter uv.exe -Recurse | Select-Object -First 1
    Copy-Item $binary.FullName $uv
    Remove-Item $archive
    Remove-Item $expanded -Recurse -Force
}
$env:NEUROSTUDIO_UV = $uv
& $uv run --no-project --python 3.12 (Join-Path $PSScriptRoot 'bootstrap.py') @args
exit $LASTEXITCODE
