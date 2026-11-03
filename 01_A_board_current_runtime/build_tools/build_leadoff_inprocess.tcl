# Build the button-enabled overlay in one Vivado process. The normal
# launch_runs flow cannot start mapped-drive child processes in this host.
if {[llength $argv] != 2} {
    error "Usage: build_leadoff_inprocess.tcl <project.xpr> <output-dir>"
}
set project_file [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
file mkdir $output_dir
open_project $project_file
set project_dir [get_property DIRECTORY [current_project]]
set_property ip_repo_paths [list [file join $project_dir ip_repo_video] \
    [file join $project_dir ip_repo_ecg] [file join $project_dir ip_repo_nn]] \
    [current_project]
update_ip_catalog -rebuild
set bd_file [get_files base.bd]
open_bd_design $bd_file
if {![llength [get_bd_cells -quiet btns_gpio]]} {
    error "The project must contain the four-button AXI GPIO"
}
validate_bd_design
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set adc_source [file join $project_dir ip_repo_ecg ads1298_acquire_ip src ads1298_acquire_ip.sv]
set adc_generated [glob -nocomplain [file join $project_dir ecg_nn_r9.gen \
    sources_1 bd base ipshared * src ads1298_acquire_ip.sv]]
if {[llength $adc_generated] != 1} {
    error "Expected one generated ADS1298 RTL copy: $adc_generated"
}
file copy -force $adc_source [lindex $adc_generated 0]
set wrappers [make_wrapper -files $bd_file -top]
if {[llength $wrappers]} {add_files -norecurse $wrappers}
set_property top base_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "LEADOFF_SYNTH_START"
synth_design -top base_wrapper -part xc7z020clg400-1 -control_set_opt_threshold 8
write_checkpoint -force [file join $output_dir synth.dcp]
puts "LEADOFF_SYNTH_COMPLETE"
opt_design
place_design
phys_opt_design
route_design -directive Explore
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
report_timing_summary -file [file join $output_dir timing_summary.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
report_drc -file [file join $output_dir drc.rpt]
write_checkpoint -force [file join $output_dir routed.dcp]
puts "LEADOFF_ROUTE_COMPLETE"
write_bitstream -force [file join $output_dir ecg_frontend.bit]
set hwh_bundle [file join $project_dir ecg_nn_r9.gen sources_1 bd base hw_handoff base.hwh]
if {![file exists $hwh_bundle]} {error "Missing HWH bundle: $hwh_bundle"}
file copy -force $hwh_bundle [file join $output_dir base_hwh_bundle.zip]
puts "LEADOFF_BIT_COMPLETE"
close_project
