$files = @(
    "GMPatel_40_sales.txt",
    "Nagraj_67_sales.txt",
    "swaminarayan_65_sales.txt"
)

$baseDir = "c:\dot_prediction_system\data"

foreach ($file in $files) {
    try {
        $inputFile = Join-Path $baseDir $file
        $outputFile = Join-Path $baseDir ($file -replace '\.txt$', '.csv')

        Write-Host "Processing $file..."

        if (-not (Test-Path $inputFile)) {
            Write-Host "  Input file not found: $inputFile"
            continue
        }

        $data = Import-Csv -Path $inputFile -Delimiter "`t"

        $results = $data | ForEach-Object {
            try {
                $val = [double]($_.Value -replace ' Kg\.$','')
            } catch {
                $val = 0
            }

            [PSCustomObject]@{
                Time        = $_.Time
                Value       = $val
                State       = $_.State
                Quality     = $_.Quality
                Reason      = $_.Reason
                Status      = $_.Status
                Suppression = $_."Suppression Type"
                Type        = "" 
            }
        }

        # Export to CSV
        $results | Export-Csv -Path $outputFile -NoTypeInformation -Encoding UTF8

        Write-Host "  Conversion complete. Output saved to $outputFile"
        Write-Host "  Total records processed: $(($results).Count)"
        Write-Host ""
    } catch {
        Write-Error "An error occurred processing $file : $_"
    }
}
