set_param general.maxThreads 4
foreach variant {reference prefetch} {
 create_project -in_memory -part xc7z020clg400-1
 if {$variant eq "reference"} {
  read_verilog -sv D:/ECG-fpga/bld/ecg_nn_r9_tft_lean/ip_repo_ecg/ecg_packet_ip/src/ecg_sync_fifo.sv
  set top ecg_sync_fifo
 } else {
  read_verilog -sv D:/ECG-fpga/pl/packet_fifo_prefetch/ecg_sync_fifo_prefetch.sv
  set top ecg_sync_fifo_prefetch
 }
 synth_design -mode out_of_context -top $top -part xc7z020clg400-1 -generic WIDTH=512 -generic DEPTH=64
 report_utilization -file D:/ECG-fpga/pl/packet_fifo_prefetch/${variant}_util.rpt
 close_project
}
exit
