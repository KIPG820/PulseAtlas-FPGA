# Add the PYNQ-Z2 four-button AXI GPIO to an opened R9 Vivado project.
# The R9 project's ecg_system.xdc already constrains BTN0..BTN3 to the
# PYNQ-Z2 pins D19, D20, L20 and L19 (btns_4bits_tri_i[3:0]).

set bd_file [get_files -quiet [file normalize [file join [get_property DIRECTORY [current_project]] \
    [file tail [file rootname [get_property NAME [current_project]]]].srcs sources_1 bd base base.bd]]]
if {[llength $bd_file] == 0} {
    set matches [get_files -all -quiet */sources_1/bd/base/base.bd]
    if {[llength $matches] != 1} {
        error "Could not uniquely locate the R9 base.bd: $matches"
    }
    set bd_file [lindex $matches 0]
} else {
    set bd_file [lindex $bd_file 0]
}

open_bd_design $bd_file

if {[llength [get_bd_cells -quiet btns_gpio]] != 0 ||
    [llength [get_bd_intf_ports -quiet btns_4bits]] != 0} {
    error "Button GPIO already exists; refusing to add a duplicate."
}

# The existing ecg_ctrl_0 SmartConnect already feeds the PS-visible R9 control
# register map. Grow it by one target while keeping the ECG/NN targets intact.
set_property CONFIG.NUM_MI {10} [get_bd_cells ecg_ctrl_0]

set btns_4bits [create_bd_intf_port -mode Master \
    -vlnv xilinx.com:interface:gpio_rtl:1.0 btns_4bits]
set btns_gpio [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:axi_gpio:2.0 btns_gpio]
set_property -dict [list \
    CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_GPIO_WIDTH {4} \
    CONFIG.C_INTERRUPT_PRESENT {0}] $btns_gpio

connect_bd_intf_net -intf_net btns_4bits_GPIO \
    [get_bd_intf_ports btns_4bits] [get_bd_intf_pins btns_gpio/GPIO]
connect_bd_intf_net -intf_net ecg_ctrl_btns_axi \
    [get_bd_intf_pins ecg_ctrl_0/M09_AXI] [get_bd_intf_pins btns_gpio/S_AXI]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins btns_gpio/s_axi_aclk]
connect_bd_net [get_bd_pins rst_ps7_0_fclk0/peripheral_aresetn] \
    [get_bd_pins btns_gpio/s_axi_aresetn]

assign_bd_address -offset 0x40007000 -range 0x00001000 \
    -target_address_space [get_bd_addr_spaces ps7_0/Data] \
    [get_bd_addr_segs btns_gpio/S_AXI/Reg]

validate_bd_design
save_bd_design
generate_target all $bd_file
make_wrapper -files $bd_file -top -import
set_property top base_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "PYNQ-Z2 four-button GPIO added at PS address 0x40007000."
