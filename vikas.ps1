param(
    [Parameter(Mandatory = $true)]
    [string] $viserver,
    [Parameter(Mandatory = $true)]
    [string] $viusername,
    [Parameter(Mandatory = $true)]
    [string] $vipassword,
    [Parameter(Mandatory = $true)]
    [string] $cluster,
    [Parameter(Mandatory = $true)]
    [string] $datastore,
    [Parameter(Mandatory = $true)]
    [string] $disk,
    [Parameter(Mandatory = $true)]
    [string] $memory,
    [Parameter(Mandatory = $true)]
    [string] $template,
    [Parameter(Mandatory = $true)]
    [string] $CPU,
    [Parameter(Mandatory = $true)]
    [string] $name
)
 
$ErrorActionPreference = "Stop"
 
Set-PowerCLIConfiguration -InvalidCertificateAction Ignore -Confirm:$False
Set-PowerCLIConfiguration -Scope User -ParticipateInCeip $False -Confirm:$False
 
Connect-VIServer -Server $viserver -User $viusername -Password $vipassword -Force
 
try {
    $vm = New-VM -Template $template -Name $name -ResourcePool $cluster -Datastore $datastore -DiskStorageFormat Thin -Confirm:$false | Set-VM -NumCpu $CPU -MemoryGB $memory -Confirm:$False
 
    Set-VM -VM $vm -MemoryHotAddEnabled $true -CpuHotAddEnabled $true -Confirm:$False
 
    $CurrentSize= Get-HardDisk -VM $vm | Where-Object {$_.Name -eq "Hard disk 1"}
    $newSize= ($currentSize.CapacityGB + $disk) 
    if($disk -ne "Null"){
        Get-HardDisk -VM $vm | Where-Object {$_.Name -eq "Hard disk 1"} | Set-HardDisk -CapacityGB $newSize -Confirm:$False
    }
 
    if($disk){
    Start-VM -VM $vm
    }
}
catch {
    Write-Host "An Error occured while creating VM $name : $($_.Exception.Message)"
}
 
# Disconnect from vCenter
Disconnect-VIServer -Server $viserver -Force -Confirm:$False
