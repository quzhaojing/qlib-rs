#Requires -Version 5.1
<#
.SYNOPSIS
Preview or delete explicitly allowlisted obsolete Qlib Rust build caches on C:.
.DESCRIPTION
Default is read-only. -Execute enables permanent deletion, with confirmation per
directory. Source, skills, Python environments, saved coverage reports and current
D: build targets are never selected. Deleted binaries can be rebuilt; they are
not sent to the Recycle Bin. Run only after stopping any jobs using the old caches.
.EXAMPLE
pwsh -File .\scripts\clean-old-c-drive-builds.ps1
.EXAMPLE
pwsh -File .\scripts\clean-old-c-drive-builds.ps1 -Execute -WhatIf
.EXAMPLE
pwsh -File .\scripts\clean-old-c-drive-builds.ps1 -Execute
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param([switch]$Execute)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Intentionally fixed allowlist: no wildcards, user-profile expansion or root deletion.
$cacheRoot = 'C:\Users\andy\.codex\tmp'
$cachePlans = @(
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-checkpoint-format-probe'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-checkpoint-format-probe'
    },
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-rs-main-target\debug'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-rs-main-target'
    },
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-rs-full-cov\llvm-cov-target\debug'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-rs-full-cov'
    },
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-rs-lossless-cov\llvm-cov-target\debug'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-rs-lossless-cov'
    },
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-rs-training-seeds-nightly\llvm-cov-target\debug'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-rs-training-seeds-nightly'
    },
    [pscustomobject]@{
        Path = 'C:\Users\andy\.codex\tmp\qlib-rs-obsolete-nightly-artifacts-20260903'
        UsageRoot = 'C:\Users\andy\.codex\tmp\qlib-rs-obsolete-nightly-artifacts-20260903'
    }
)

function Assert-CachePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $absolutePath = [IO.Path]::GetFullPath($Path)
    if ($absolutePath -cne $Path -or
        -not $absolutePath.StartsWith($cacheRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $Path -notin $cachePlans.Path) {
        throw "Refusing path outside the exact cache allowlist: $Path"
    }
    # Check every existing ancestor, not just the leaf: a parent junction could
    # otherwise redirect a lexically safe path onto another directory or drive.
    $ancestor = $absolutePath
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing junction/symlink/reparse ancestor: $ancestor"
            }
            if (-not $item.PSIsContainer) { throw "Expected directory: $ancestor" }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
}

function Get-CacheBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    # Explicit traversal checks each entry BEFORE descending, even on older
    # PowerShell versions whose recursive symlink behavior may differ.
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($Path)
    [long]$bytes = 0
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($entry in Get-ChildItem -LiteralPath $directory -Force) {
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing nested junction/symlink/reparse entry: $($entry.FullName)"
            }
            if ($entry.PSIsContainer) { $pending.Push($entry.FullName) }
            else { $bytes += $entry.Length }
        }
    }
    return $bytes
}

function Assert-CacheIdle {
    param([Parameter(Mandatory = $true)][string]$UsageRoot)

    # Failing to enumerate processes is an error, not evidence that caches are idle.
    $users = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and (
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith(
                $UsageRoot + '\', [StringComparison]::OrdinalIgnoreCase)) -or
            ($_.CommandLine -and $_.CommandLine.IndexOf(
                $UsageRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0)
        )
    })
    if ($users.Count -gt 0) {
        $details = ($users | ForEach-Object { "$($_.Name) PID=$($_.ProcessId)" }) -join ', '
        throw "Cache appears in a live process: $UsageRoot ($details). Stop that job first."
    }
}

$freeBefore = (Get-PSDrive -Name C).Free
$inventory = @()
Write-Host 'Inspecting obsolete caches. No files are deleted during this scan.'
foreach ($plan in $cachePlans) {
    Assert-CachePath -Path $plan.Path
    $present = Test-Path -LiteralPath $plan.Path
    [long]$bytes = 0
    if ($present) {
        Assert-CacheIdle -UsageRoot $plan.UsageRoot
        $bytes = Get-CacheBytes -Path $plan.Path
    }
    $inventory += [pscustomobject]@{
        Path = $plan.Path
        UsageRoot = $plan.UsageRoot
        Present = $present
        Bytes = $bytes
        GiB = [math]::Round($bytes / 1GB, 3)
    }
}
$inventory | Select-Object Path, Present, GiB | Format-Table -AutoSize -Wrap
$totalBytes = ($inventory | Measure-Object -Property Bytes -Sum).Sum
Write-Host ('Logical cache size: {0:N3} GiB. C: free: {1:N3} GiB.' -f ($totalBytes / 1GB), ($freeBefore / 1GB))
Write-Host 'Actual recovered disk space can differ due to compression or hard links.'

if (-not $Execute) {
    Write-Host 'PREVIEW ONLY. Rerun with -Execute to delete, or -Execute -WhatIf to simulate.'
    return
}

Write-Warning 'Permanent deletion of rebuildable artifacts only. No Recycle Bin recovery.'
$removed = 0
try {
    foreach ($cache in $inventory) {
        if (-not $cache.Present) { continue }
        if ($PSCmdlet.ShouldProcess($cache.Path, 'Permanently delete obsolete Rust build cache')) {
            # Revalidate after any interactive confirmation. Do not start old builds
            # concurrently: a process can still start between this check and deletion.
            Assert-CachePath -Path $cache.Path
            Assert-CacheIdle -UsageRoot $cache.UsageRoot
            $null = Get-CacheBytes -Path $cache.Path
            Remove-Item -LiteralPath $cache.Path -Recurse -Force -Confirm:$false -ErrorAction Stop
            if (Test-Path -LiteralPath $cache.Path) { throw "Directory remains after deletion: $($cache.Path)" }
            $removed++
            Write-Host "Removed: $($cache.Path)"
        }
    }
}
finally {
    $freeAfter = (Get-PSDrive -Name C).Free
    Write-Host ('Removed {0} directories; measured C: free-space change: {1:N3} GiB; now free: {2:N3} GiB.' -f
        $removed, (($freeAfter - $freeBefore) / 1GB), ($freeAfter / 1GB))
    Write-Host 'Source, skills, Python oracle, saved coverage reports and D: targets were not selected.'
}
