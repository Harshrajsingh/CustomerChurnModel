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

function Get-SonarSeverity($severity) {
    switch ($severity) {
        "Error"       { return "CRITICAL" }
        "Warning"     { return "MAJOR" }
        "Information" { return "INFO" }
        "ParseError"  { return "BLOCKER" }
        default       { return "MINOR" }
    }
}

function Get-SonarType($severity) {
    switch ($severity) {
        "Error"       { return "BUG" }
        "ParseError"  { return "BUG" }
        "Warning"     { return "CODE_SMELL" }
        "Information" { return "CODE_SMELL" }
        default       { return "CODE_SMELL" }
    }
}

# Collect unique rules (fixes the deprecation warning)
$rules = $results | Select-Object -ExpandProperty RuleName -Unique | ForEach-Object {
    @{
        id                = $_
        name              = $_
        description       = "PSScriptAnalyzer rule: $_"
        engineId          = "PSScriptAnalyzer"
        cleanCodeAttribute = "CONVENTIONAL"
        impacts           = @(
            @{
                softwareQuality = "MAINTAINABILITY"
                severity        = "MEDIUM"
            }
        )
    }
}

$issues = $results | ForEach-Object {
    # Make path relative with forward slashes
    $relativePath = $_.ScriptPath -replace [regex]::Escape($resolvedSource + "\"), ""
    $relativePath = $relativePath -replace "\\", "/"

    # FIX: SonarQube requires line >= 1, default to 1 if 0 or null
    $lineNumber = if ($_.Line -gt 0) { [int]$_.Line } else { 1 }

    @{
        engineId        = "PSScriptAnalyzer"
        ruleId          = $_.RuleName
        severity        = Get-SonarSeverity($_.Severity.ToString())
        type            = Get-SonarType($_.Severity.ToString())
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
    rules  = $rules      # fixes deprecation warning
    issues = $issues     # fixes line 0 error
}

$sonarReport | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputFile -Encoding UTF8

Write-Host "=== Report saved to: $OutputFile ==="
Write-Host "=== Total issues: $($issues.Count) ==="
