set here [file normalize [file dirname [info script]]]
set ip_root [file dirname $here]
set out_dir [file join $ip_root packaged]

create_project -in_memory -part xc7z020clg400-1
add_files -norecurse [list \
    [file join $ip_root rtl ads1298_spi_master.sv] \
    [file join $ip_root rtl ads1298_acquire_ip.sv]]
set_property top ads1298_acquire_ip [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $out_dir -vendor ecg.local \
    -library user -taxonomy /UserIP -import_files -force
set core [ipx::current_core]
set_property name ads1298_acquire_ip $core
set_property version 1.0 $core
set_property display_name {ADS1298 ECG Acquisition} $core
set_property description {8-channel 24-bit ADS1298 SPI acquisition for PYNQ-Z2} $core

set axi [ipx::get_bus_interfaces s_axi -of_objects $core]
set protocol [ipx::add_bus_parameter PROTOCOL $axi]
set_property value AXI4LITE $protocol
set clock_if [ipx::get_bus_interfaces aclk -of_objects $core]
set freq [ipx::add_bus_parameter FREQ_HZ $clock_if]
set_property value 100000000 $freq

ipx::save_core $core
puts "PACKAGED_IP=[get_property xml_file_name $core]"
