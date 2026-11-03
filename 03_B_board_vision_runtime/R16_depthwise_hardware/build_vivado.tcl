# Build a separate B-board PS + vision-only overlay from the validated PYNQ-Z2
# PS configuration. The A/ECG and R3 vision projects are left untouched.
set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. ..]]
set base_dir [file join $root PYNQ-3.1.1 boards Pynq-Z2 base_2024_2]
set pynq_ip_repo [file join $root PYNQ-3.1.1 boards ip]
set vision_ip_repo [file join $root b_board PulseAtlas_B_Vision_R16_Depthwise vision_accel ip_repo catalog]
set project_name vision_bonly_depthwise_95mhz
set project_dir [file join $here vivado]
set out_dir [file join $here vision_accel]

if {![file exists [file join $base_dir base.tcl]]} { error "Missing PYNQ-Z2 base.tcl" }
if {![file exists [file join $vision_ip_repo component.xml]]} { error "Missing line-cache HLS IP" }

create_project -force $project_name $project_dir -part xc7z020clg400-1
set_property ip_repo_paths [list $pynq_ip_repo $vision_ip_repo] [current_project]

# Reuse the board-tested PS/DDR configuration from PYNQ 3.1.1, then prune the
# base design down to the PS DDR/FIXED_IO ports and the new vision engine.
cd $base_dir
source [file join $base_dir base.tcl]
current_bd_design base

foreach cell [get_bd_cells -quiet] {
    if {[get_property NAME $cell] ne "ps7_0"} { delete_bd_objs $cell }
}
foreach port [get_bd_intf_ports -quiet] {
    if {[get_property NAME $port] ni {DDR FIXED_IO}} { delete_bd_objs $port }
}
foreach port [get_bd_ports -quiet] {
    set name [get_property NAME $port]
    if {![string match "DDR_*" $name] && ![string match "FIXED_IO_*" $name]} {
        delete_bd_objs $port
    }
}
# The reference base BD wired several PS clock pins to other clock nets. Remove
# the old internal nets after pruning so the reduced design can reconnect them
# cleanly to FCLK_CLK0 and its reset domain.
delete_bd_objs [get_bd_nets -quiet]

set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_GP0 {0} \
    CONFIG.PCW_USE_S_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_USE_S_AXI_HP2 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_S_AXI_HP1_DATA_WIDTH {64} \
    CONFIG.PCW_S_AXI_HP2_DATA_WIDTH {64} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_CLK0_FREQ {95000000} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {95.000000} \
] [get_bd_cells ps7_0]

set_property ip_repo_paths [list $pynq_ip_repo $vision_ip_repo] [current_project]
update_ip_catalog -rebuild
set ipdef [get_ipdefs -all -quiet pulseatlas.local:vision:vision_conv_writebase_v19:1.0]
if {[llength $ipdef] != 1} { error "Expected one line-cache IP, found [llength $ipdef]" }

create_bd_cell -type ip -vlnv pulseatlas.local:vision:vision_conv_writebase_v19:1.0 vision_conv_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 vision_ctrl_interconnect
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells vision_ctrl_interconnect]
foreach port {ACT PARAM OUT_r} hp {0 1 2} {
    set interconnect vision_mem_${port}_interconnect
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 $interconnect
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells $interconnect]
    connect_bd_intf_net [get_bd_intf_pins vision_conv_0/m_axi_${port}] \
        [get_bd_intf_pins ${interconnect}/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins ${interconnect}/M00_AXI] \
        [get_bd_intf_pins ps7_0/S_AXI_HP${hp}]
}
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_vision_fclk0
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 vision_dcm_locked
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells vision_dcm_locked]

connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] \
    [get_bd_intf_pins vision_ctrl_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins vision_ctrl_interconnect/M00_AXI] \
    [get_bd_intf_pins vision_conv_0/s_axi_CTRL]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] \
    [get_bd_pins vision_conv_0/ap_clk] \
    [get_bd_pins vision_ctrl_interconnect/ACLK] \
    [get_bd_pins vision_ctrl_interconnect/S00_ACLK] \
    [get_bd_pins vision_ctrl_interconnect/M00_ACLK] \
    [get_bd_pins vision_mem_ACT_interconnect/aclk] \
    [get_bd_pins vision_mem_PARAM_interconnect/aclk] \
    [get_bd_pins vision_mem_OUT_r_interconnect/aclk] \
    [get_bd_pins ps7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins ps7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins ps7_0/S_AXI_HP1_ACLK] \
    [get_bd_pins ps7_0/S_AXI_HP2_ACLK] \
    [get_bd_pins rst_vision_fclk0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] \
    [get_bd_pins rst_vision_fclk0/ext_reset_in]
connect_bd_net [get_bd_pins vision_dcm_locked/dout] \
    [get_bd_pins rst_vision_fclk0/dcm_locked]
connect_bd_net [get_bd_pins rst_vision_fclk0/peripheral_aresetn] \
    [get_bd_pins vision_conv_0/ap_rst_n] \
    [get_bd_pins vision_ctrl_interconnect/ARESETN] \
    [get_bd_pins vision_ctrl_interconnect/S00_ARESETN] \
    [get_bd_pins vision_ctrl_interconnect/M00_ARESETN] \
    [get_bd_pins vision_mem_ACT_interconnect/aresetn] \
    [get_bd_pins vision_mem_PARAM_interconnect/aresetn] \
    [get_bd_pins vision_mem_OUT_r_interconnect/aresetn]

assign_bd_address -offset 0x43C00000 -range 64K \
    [get_bd_addr_segs vision_conv_0/s_axi_CTRL/Reg]
set hp0_ddr [get_bd_addr_segs ps7_0/S_AXI_HP0/HP0_DDR_LOWOCM]
set hp1_ddr [get_bd_addr_segs ps7_0/S_AXI_HP1/HP1_DDR_LOWOCM]
set hp2_ddr [get_bd_addr_segs ps7_0/S_AXI_HP2/HP2_DDR_LOWOCM]
assign_bd_address -target_address_space vision_conv_0/Data_m_axi_ACT $hp0_ddr
assign_bd_address -target_address_space vision_conv_0/Data_m_axi_PARAM $hp1_ddr
assign_bd_address -target_address_space vision_conv_0/Data_m_axi_OUT_r $hp2_ddr

validate_bd_design
save_bd_design
set bd_file [get_files base.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set wrappers [make_wrapper -files $bd_file -top -force]
if {[llength $wrappers] > 0} { add_files -norecurse $wrappers }
set_property top base_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "VISION_BONLY_SYNTH_STATUS=$synth_status"
if {![string match "*Complete*" $synth_status]} { error "Synthesis failed: $synth_status" }
open_run synth_1
report_utilization -hierarchical -file [file join $out_dir vision_bonly_depthwise_95mhz_synth_util.rpt]
report_timing_summary -delay_type min_max -file [file join $out_dir vision_bonly_depthwise_95mhz_synth_timing.rpt]
close_design

# 95 MHz closure: explore placement and perform pre-route physical optimization.
# These are implementation choices; no clock or timing exception is relaxed.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "VISION_BONLY_ROUTE_STATUS=$impl_status"
if {![string match "*Complete*" $impl_status]} { error "Implementation failed before route completion: $impl_status" }
open_run impl_1
report_utilization -hierarchical -file [file join $out_dir vision_bonly_depthwise_95mhz_route_util.rpt]
report_timing_summary -delay_type min_max -file [file join $out_dir vision_bonly_depthwise_95mhz_route_timing.rpt]
report_route_status -file [file join $out_dir vision_bonly_depthwise_95mhz_route_status.rpt]
set wns [get_property STATS.WNS [get_runs impl_1]]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {[llength $hold_paths] != 1} {error "Missing routed hold timing path"}
set whs [get_property SLACK $hold_paths]
puts "VISION_BONLY_ROUTE_WNS_NS=$wns"
puts "VISION_BONLY_ROUTE_WHS_NS=$whs"
report_drc -file [file join $out_dir vision_bonly_depthwise_95mhz_route_drc.rpt]
close_design

if {$wns >= 0.0 && $whs >= 0.0} {
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    set bit_file [file join $project_dir ${project_name}.runs impl_1 base_wrapper.bit]
    if {![file exists $bit_file]} { error "Expected routed bitstream not found: $bit_file" }
    file copy -force $bit_file [file join $out_dir vision_bonly_depthwise_95mhz.bit]
    set hwh_file [file join $project_dir ${project_name}.gen sources_1 bd base hw_handoff base.hwh]
    if {![file exists $hwh_file]} { error "Expected hardware handoff not found: $hwh_file" }
    file copy -force $hwh_file [file join $out_dir vision_bonly_depthwise_95mhz.hwh]
}

close_project
exit




