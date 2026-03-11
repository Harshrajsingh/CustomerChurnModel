# run-psscriptanalyzer.ps1
param(
    [string]$SourcePath = ".",
    [string]$OutputFile = "sonar-ps-issues.json"
)

Write-Host "=== Scanning PowerShell files in: $SourcePath ==="

$resolvedSource = (Resolve-Path $SourcePath).Path

# Get all .ps1 files and scan them
$results = Get-ChildItem -Path $resolvedSource -Recurse -Filter "*.ps1" | ForEach-Object {
    Invoke-ScriptAnalyzer -Path $_.FullName
}

if (-not $results) {
    Write-Host "No issues found."
    @{ issues = @() } | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputFile
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

$issues = $results | ForEach-Object {
    # Convert absolute path → relative path with forward slashes
    $relativePath = $_.ScriptPath -replace [regex]::Escape($resolvedSource + "\"), ""
    $relativePath = $relativePath -replace "\\", "/"

    @{
        engineId        = "PSScriptAnalyzer"
        ruleId          = $_.RuleName
        severity        = Get-SonarSeverity($_.Severity.ToString())
        type            = Get-SonarType($_.Severity.ToString())
        primaryLocation = @{
            message   = $_.Message -replace '"', "'"   # escape quotes for valid JSON
            filePath  = $relativePath
            textRange = @{
                startLine = [int]$_.Line
            }
        }
    }
}

$sonarReport = @{ issues = $issues }
$sonarReport | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputFile -Encoding UTF8

Write-Host "=== Report saved to: $OutputFile ==="
Write-Host "=== Total issues: $($issues.Count) ==="
