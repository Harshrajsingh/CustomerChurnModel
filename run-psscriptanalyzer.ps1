# run-psscriptanalyzer.ps1
param(
    [string]$SourcePath = ".",
    [string]$OutputFile = "sonar-ps-issues.json"
)

Write-Host "=== Scanning PowerShell files in: $SourcePath ==="

$resolvedSource = (Resolve-Path $SourcePath).Path

$results = Get-ChildItem -Path $resolvedSource -Recurse -Filter "*.ps1" | ForEach-Object {
    Invoke-ScriptAnalyzer -Path $_.FullName
}

if (-not $results) {
    Write-Host "No issues found."
    @{ rules = @(); issues = @() } | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputFile -Encoding UTF8
    exit 0
}

# Map severity to SonarQube impact severity (new format)
function Get-ImpactSeverity($severity) {
    switch ($severity) {
        "Error"       { return "HIGH" }
        "Warning"     { return "MEDIUM" }
        "Information" { return "LOW" }
        "ParseError"  { return "HIGH" }
        default       { return "LOW" }
    }
}

# Build unique rules list (required in new format)
$rules = $results | Select-Object -ExpandProperty RuleName -Unique | ForEach-Object {
    @{
        id                 = $_
        name               = $_
        description        = "PSScriptAnalyzer rule: $_"
        engineId           = "PSScriptAnalyzer"
        cleanCodeAttribute = "CONVENTIONAL"
        impacts            = @(
            @{
                softwareQuality = "MAINTAINABILITY"
                severity        = "MEDIUM"
            }
        )
    }
}

# Build issues list — NO severity, NO type in new format
$issues = $results | ForEach-Object {
    # Convert absolute path to relative with forward slashes
    $relativePath = $_.ScriptPath -replace [regex]::Escape($resolvedSource + "\"), ""
    $relativePath = $relativePath -replace "\\", "/"

    # SonarQube requires line >= 1
    $lineNumber = if ($_.Line -gt 0) { [int]$_.Line } else { 1 }

    @{
        engineId        = "PSScriptAnalyzer"
        ruleId          = $_.RuleName
        # ← NO severity field here (forbidden in new format)
        # ← NO type field here (forbidden in new format)
        impacts         = @(
            @{
                softwareQuality = "MAINTAINABILITY"
                severity        = Get-ImpactSeverity($_.Severity.ToString())
            }
        )
        primaryLocation = @{
            message   = $_.Message -replace '"', "'"
            filePath  = $relativePath
            textRange = @{
                startLine = $lineNumber
            }
        }
    }
}

$sonarReport = @{
    rules  = $rules
    issues = $issues
}

$sonarReport | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputFile -Encoding UTF8

Write-Host "=== Report saved to: $OutputFile ==="
Write-Host "=== Total issues: $($issues.Count) ==="
