param(
    [Parameter(Mandatory = $true)]
    [string]$ReportPath,
    [switch]$RequireBranches
)

# Use raw LLVM counts, not rounded percentages. cargo-llvm-cov 0.9.0's
# --fail-under-file-lines compares with > rather than >=, rejecting 100% itself.
$ErrorActionPreference = 'Stop'
$coverageReport = Get-Content -Raw -LiteralPath $ReportPath | ConvertFrom-Json
if ($null -eq $coverageReport.data -or @($coverageReport.data).Count -eq 0) {
    throw 'Coverage report contains no data.'
}
$metrics = @('lines', 'functions', 'regions')
if ($RequireBranches) { $metrics += 'branches' }
$fileCount = 0
$branchCount = 0
foreach ($export in $coverageReport.data) {
    if ($null -eq $export.files -or @($export.files).Count -eq 0) { throw 'Coverage export contains no files.' }
    $scopes = @([pscustomobject]@{ name = 'TOTAL'; summary = $export.totals })
    foreach ($file in $export.files) {
        $fileCount++
        $scopes += [pscustomobject]@{ name = $file.filename; summary = $file.summary }
    }
    foreach ($scope in $scopes) {
        foreach ($metric in $metrics) {
            $counts = $scope.summary.$metric
            if ($null -eq $counts.count -or $null -eq $counts.covered) {
                throw "Missing $metric counts for $($scope.name)."
            }
            if ($counts.count -lt 0 -or $counts.covered -ne $counts.count) {
                throw "Coverage below 100%: $($scope.name) $metric $($counts.covered)/$($counts.count)."
            }
        }
    }
    if ($export.totals.lines.count -le 0) { throw 'Coverage export has no executable lines.' }
    if ($RequireBranches) { $branchCount += $export.totals.branches.count }
}
if ($RequireBranches -and $branchCount -le 0) {
    throw 'Branch coverage was required but no branches were instrumented.'
}
Write-Output "Exact 100% coverage verified for $fileCount files: $($metrics -join ', ')."
