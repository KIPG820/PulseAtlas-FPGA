# Build the R9 project after add_pynq_buttons.tcl has been applied.
# Arguments: <ecg_nn_r9.xpr> <output-directory>

if {[llength $argv] != 2} {
    error "Usage: build_button_overlay.tcl <ecg_nn_r9.xpr> <output-directory>"
}
set project_file [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
file mkdir $output_dir
open_project $project_file

# Vivado keeps a generated ipshared copy of packaged RTL. Refresh both the
# catalog and that copy before resetting runs, otherwise a new bitstream can
# silently contain the old ADC implementation.
set project_dir [get_property DIRECTORY [current_project]]
set_property ip_repo_paths [list [file join $project_dir ip_repo_video] \
    [file join $project_dir ip_repo_ecg] [file join $project_dir ip_repo_nn]] \
    [current_project]
update_ip_catalog -rebuild

set bd_path [file join [get_property DIRECTORY [current_project]] \
    [file tail [file rootname [get_property NAME [current_project]]]].srcs \
    sources_1 bd base base.bd]
open_bd_design [get_files [file normalize $bd_path]]
set button_cell [get_bd_cells -quiet btns_gpio]
if {[llength $button_cell] == 0} {
    error "btns_gpio is missing. Run add_pynq_buttons.tcl and regenerate the block design first."
}

set adc_source [file join $project_dir ip_repo_ecg ads1298_acquire_ip src ads1298_acquire_ip.sv]
generate_target all [get_files [file normalize $bd_path]]
set adc_generated [glob -nocomplain [file join $project_dir ecg_nn_r9.gen \
    sources_1 bd base ipshared * src ads1298_acquire_ip.sv]]
if {[llength $adc_generated] != 1} {
    error "Expected one generated ADS1298 RTL copy: $adc_generated"
}
file copy -force $adc_source [lindex $adc_generated 0]
reset_run impl_1
reset_run synth_1
if {[llength [get_runs -quiet base_ecg_adc_0_synth_1]]} {
    reset_run base_ecg_adc_0_synth_1
}

update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "R9 button synthesis status: $synth_status"
if {![string match "*Complete*" $synth_status]} {
    error "R9 button synthesis did not complete successfully."
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "R9 button implementation status: $impl_status"
if {![string match "*Complete*" $impl_status]} {
    error "R9 button implementation / bitstream generation did not complete."
}

set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bit_file [file join $impl_dir ecg_nn_r9.bit]
if {![file exists $bit_file]} {
    error "Expected bitstream not found: $bit_file"
}
file copy -force $bit_file [file join $output_dir ecg_frontend.bit]

# Vivado 2024.2's HWH handoff is a zip bundle containing the plain PYNQ HWH.
set hwh_bundle [file join [get_property DIRECTORY [current_project]] \
    ecg_nn_r9.gen sources_1 bd base hw_handoff base.hwh]
if {![file exists $hwh_bundle]} {
    error "Expected generated HWH bundle not found: $hwh_bundle"
}
file copy -force $hwh_bundle [file join $output_dir base_hwh_bundle.zip]

open_run impl_1
report_timing_summary -file [file join $output_dir timing_summary.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
close_project
puts "Button-enabled bitstream and HWH bundle written to $output_dir"
