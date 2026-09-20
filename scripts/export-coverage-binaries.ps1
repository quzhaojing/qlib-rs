param(
    [Parameter(Mandatory = $true)][string]$LlvmCovPath,
    [Parameter(Mandatory = $true)][string]$ProfilePath,
    [Parameter(Mandatory = $true)][string[]]$BinaryPath,
    [Parameter(Mandatory = $true)][string]$ReportPath
)

# Export only the executables actually run by the recorded cargo-llvm-cov test
# invocation. Never discover all cached binaries: older crate versions can carry
# obsolete coverage maps. Include EVERY executable run for the requested scope.
# This script does not run tests, filter sources, merge profiles, or alter counts.
$ErrorActionPreference = 'Stop'
$tool = (Get-Item -LiteralPath $LlvmCovPath -ErrorAction Stop).FullName
$profile = (Get-Item -LiteralPath $ProfilePath -ErrorAction Stop).FullName
$binaries = @($BinaryPath | ForEach-Object {
    (Get-Item -LiteralPath $_ -ErrorAction Stop).FullName
})
if ($binaries.Count -eq 0) { throw 'At least one actual test executable is required.' }
$report = [IO.Path]::GetFullPath($ReportPath)
$manifest = "$report.inputs.json"
if ((Test-Path -LiteralPath $report) -or (Test-Path -LiteralPath $manifest)) {
    throw 'Refusing to overwrite an existing report or input manifest.'
}
if (-not (Test-Path -LiteralPath ([IO.Path]::GetDirectoryName($report)) -PathType Container)) {
    throw 'The output directory must already exist.'
}
$inputs = @($tool, $profile) + $binaries
$before = @($inputs | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash)
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $tool
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.ArgumentList.Add('export')
$start.ArgumentList.Add($binaries[0])
$start.ArgumentList.Add("-instr-profile=$profile")
foreach ($binary in $binaries | Select-Object -Skip 1) {
    $start.ArgumentList.Add('-object')
    $start.ArgumentList.Add($binary)
}
$process = [Diagnostics.Process]::new()
$process.StartInfo = $start
$stream = [IO.File]::Open($report, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    if (-not $process.Start()) { throw 'LLVM coverage exporter did not start.' }
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.StandardOutput.BaseStream.CopyTo($stream)
    $process.WaitForExit()
    $diagnostics = $stderr.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
        throw "LLVM export failed ($($process.ExitCode)); partial report retained. $diagnostics"
    }
    if ($diagnostics) { Write-Warning $diagnostics }
} finally {
    $stream.Dispose()
    $process.Dispose()
}
$after = @($inputs | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash)
if (Compare-Object $before $after -Property Path, Hash) {
    throw 'Coverage input changed during export; report is not valid.'
}
$raw = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
if ($raw.type -ne 'llvm.coverage.json.export' -or @($raw.data.files).Count -eq 0) {
    throw 'LLVM export contains no file coverage; report retained but not accepted.'
}
$metadata = [ordered]@{
    inputs = $before
    arguments = @($start.ArgumentList)
    report = (Get-FileHash -LiteralPath $report -Algorithm SHA256 | Select-Object Path, Hash)
    note = 'Unmodified LLVM export; no source exclusions or coverage acceptance implied.'
} | ConvertTo-Json -Depth 5
$metadataStream = [IO.File]::Open($manifest, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($metadata)
    $metadataStream.Write($bytes, 0, $bytes.Length)
} finally {
    $metadataStream.Dispose()
}
Write-Output "Raw coverage report: $report"
Write-Output "Input hashes and exact argv: $manifest"
