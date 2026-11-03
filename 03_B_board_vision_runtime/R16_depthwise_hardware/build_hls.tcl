set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. ..]]
set src [file join $root nn pl vision_shared_hls experiments im2col_linecache_v20_depthwise]
set workspace [file join $root bld vision_bonly_depthwise_v20]
file mkdir $workspace
cd $workspace
open_project -reset candidate
set_top vision_conv_writebase_v19
add_files [file join $src vision_conv_writebase_v19.cpp] -cflags "-std=c++14 -DPULSEATLAS_OC_BLOCKS=2"
add_files -tb [file join $src vision_conv_writebase_v19_tb.cpp] -cflags "-std=c++14"
open_solution -reset solution1
set_part xc7z020clg400-1
create_clock -period 11 -name default
csim_design
csynth_design
exit
