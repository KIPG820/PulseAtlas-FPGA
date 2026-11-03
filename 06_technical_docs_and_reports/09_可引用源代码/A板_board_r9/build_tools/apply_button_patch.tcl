if {[llength $argv] != 1} {
    error "Usage: apply_button_patch.tcl <ecg_nn_r9.xpr>"
}
set project_file [file normalize [lindex $argv 0]]
set project_dir [file dirname $project_file]
open_project $project_file
source [file join [file dirname [info script]] add_pynq_buttons.tcl]
set hwh_file [file join $project_dir ecg_nn_r9.gen sources_1 bd base hw_handoff base.hwh]
file mkdir [file dirname $hwh_file]
write_hwdef -force -file $hwh_file
puts "Updated HWH: $hwh_file"
close_project
