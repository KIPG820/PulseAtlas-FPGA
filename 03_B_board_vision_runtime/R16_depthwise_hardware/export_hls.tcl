set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. ..]]
set ip_repo [file join $here vision_accel ip_repo]
file mkdir $ip_repo
cd [file join $root bld vision_bonly_depthwise_v20]
open_project candidate
open_solution solution1
export_design -format ip_catalog -rtl verilog -output $ip_repo -vendor pulseatlas.local -library vision -version 1.0
set catalog_dir [file join $ip_repo catalog]
file mkdir $catalog_dir
exec powershell.exe -NoProfile -Command "Expand-Archive -LiteralPath '$ip_repo/export.zip' -DestinationPath '$catalog_dir' -Force"
exit
