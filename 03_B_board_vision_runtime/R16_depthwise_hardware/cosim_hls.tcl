set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. ..]]
cd [file join $root bld vision_bonly_depthwise_v20]
open_project candidate
open_solution solution1
cosim_design -rtl verilog
exit
